"""Prometheus metrics exposed by the gateway on /metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---- HTTP
HTTP_REQUESTS = Counter(
    "adaptos_http_requests_total",
    "HTTP requests handled by the gateway",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "adaptos_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

# ---- Routing and generation (label "adapter" is a task name or "base")
ROUTED_REQUESTS = Counter(
    "adaptos_routed_requests_total",
    "Chat requests by the adapter that answered and the routing reason",
    ["adapter", "reason"],
)
ROUTER_CONFIDENCE = Histogram(
    "adaptos_router_confidence",
    "Router confidence of the top label",
    ["label"],
    buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
GENERATION_LATENCY = Histogram(
    "adaptos_generation_seconds",
    "Time spent generating an answer",
    ["adapter"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
OUTPUT_TOKENS = Counter("adaptos_output_tokens_total", "Generated tokens", ["adapter"])
FEEDBACK = Counter("adaptos_feedback_total", "User feedback", ["adapter", "rating"])

# ---- Online quality and rolling-window health (refreshed by the monitor loop)
QUALITY_SCORE = Histogram(
    "adaptos_quality_score",
    "Reference-free quality score of each answer",
    ["adapter"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
WINDOW_QUALITY = Gauge(
    "adaptos_window_quality", "Mean quality over the adapter's recent window", ["adapter"]
)
WINDOW_DRIFT = Gauge(
    "adaptos_window_drift", "Feature-drift score of the adapter's recent window", ["adapter"]
)
WINDOW_DRIFT_FEATURE = Gauge(
    "adaptos_window_drift_feature",
    "Standardized shift of each request feature vs the training reference",
    ["adapter", "feature"],
)
WINDOW_SAMPLES = Gauge("adaptos_window_samples", "Requests in the window", ["adapter"])
ADAPTER_ALERT = Gauge(
    "adaptos_adapter_alert", "1 when the adapter's window breaches a threshold", ["adapter"]
)
QUALITY_THRESHOLD = Gauge("adaptos_quality_threshold", "Configured quality_min", ["adapter"])
DRIFT_THRESHOLD = Gauge("adaptos_drift_threshold", "Configured drift_max", ["adapter"])

# ---- Deployed versions: 1 for the live version of each model
LIVE_VERSION = Gauge(
    "adaptos_live_model_version",
    "Registry version currently served (value is the version number)",
    ["model"],
)
MODEL_SWAPS = Counter(
    "adaptos_model_swaps_total", "Hot-swaps of a served model version", ["model", "action"]
)
