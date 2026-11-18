"""Multi-adapter inference: one base model in memory, many LoRA adapters on top.

Used by offline evaluation (Phase 3) and by the serving gateway (Phase 5). Adapters can be
added, replaced (hot-swap to a new version) and removed while the base model stays loaded.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptos.log import get_logger
from adaptos.prompts import build_messages

logger = get_logger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "cpu",
        num_threads: int | None = None,
        dtype: str | None = None,
        model_loader: Any = None,
    ):
        import torch

        self.device = device
        if num_threads and device == "cpu":
            torch.set_num_threads(num_threads)
        if model_loader is not None:
            self.tokenizer, base = model_loader()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            # Default: fp16 on GPU, fp32 on CPU. Serving passes bfloat16 (measured faster on CPU).
            torch_dtype = (
                getattr(torch, dtype)
                if dtype
                else (torch.float16 if device == "cuda" else torch.float32)
            )
            base = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=torch_dtype
            )
        self.tokenizer.padding_side = "left"  # decoder-only models generate after the prompt
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = base.to(device).eval()
        self.model: Any = self.base
        self.adapters: dict[str, str] = {}  # adapter name -> source path
        self.lock = threading.Lock()  # generation and adapter swaps share one model
        self.model_id, self.revision = model_id, revision

    # ---------------------------------------------------------------- adapters
    def load_adapter(self, name: str, path: Path | str) -> None:
        """Add an adapter, or replace it if ``name`` is already loaded (hot-swap)."""
        from peft import PeftModel

        with self.lock:
            if not self.adapters:
                self.model = PeftModel.from_pretrained(self.base, str(path), adapter_name=name)
            else:
                if name in self.adapters:
                    self.model.delete_adapter(name)
                self.model.load_adapter(str(path), adapter_name=name)
            self.model.eval()
            self.adapters[name] = str(path)
        logger.info("adapter '%s' loaded from %s", name, path)

    def unload_adapter(self, name: str) -> None:
        with self.lock:
            if name not in self.adapters:
                return
            if len(self.adapters) == 1:
                self.base = self.model.unload()  # back to the plain base model
                self.model = self.base
            else:
                self.model.delete_adapter(name)
            del self.adapters[name]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # -------------------------------------------------------------- generation
    def generate(
        self,
        inputs: Sequence[str],
        adapter: str | None,
        prompt_task: str | None,
        max_new_tokens: int,
        batch_size: int = 8,
    ) -> list[str]:
        """Greedy generation. ``adapter=None`` uses the plain base model; ``prompt_task`` picks
        the system prompt (so a base-model baseline can use the same prompt as the adapter)."""
        import torch

        if adapter is not None and adapter not in self.adapters:
            raise KeyError(f"adapter '{adapter}' is not loaded; loaded: {sorted(self.adapters)}")
        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(prompt_task, text), tokenize=False, add_generation_prompt=True
            )
            for text in inputs
        ]
        outputs: list[str] = []
        with self.lock, torch.no_grad():
            if adapter is None and self.adapters:
                ctx: Any = self.model.disable_adapter()
            else:
                if adapter is not None:
                    self.model.set_adapter(adapter)
                ctx = contextlib.nullcontext()
            with ctx:
                for i in range(0, len(prompts), batch_size):
                    enc = self.tokenizer(
                        prompts[i : i + batch_size],
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).to(self.device)
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,  # clear the model's sampling defaults (greedy)
                        top_p=None,
                        top_k=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    new_tokens = out[:, enc["input_ids"].shape[1] :]
                    outputs += [
                        t.strip()
                        for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    ]
        return outputs
