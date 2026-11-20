# Live Walkthrough (AWS deployment)

AdaptOS runs on AWS: one GPU server (g4dn.xlarge) with the full stack and the same trained models as the development laptop. Every link below opens from any browser, anywhere, with **no login**.

> **This deployment is temporary.** The server deletes itself on **20 Nov 2026 at 5:22 PM IST (11:52 UTC)**, at the latest, to keep the cost under $5. After that these links stop working.

---

## Live links

| Service | Link |
|---|---|
| **Gateway: ask the AI** (Swagger UI) | https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/docs |
| **MLflow: models, versions, runs** | https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com |
| **Grafana: live dashboard** | https://iafg8o1nzf.execute-api.us-east-1.amazonaws.com/d/adaptos-overview |
| **Prometheus: metrics and alerts** | https://67j6ijkgl5.execute-api.us-east-1.amazonaws.com |

**Quick status pages** (open directly in the browser):

| Page | Link | Shows |
|---|---|---|
| Ready | https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/ready | models, MLflow and database are all up |
| Live versions | https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/models | which version of each model is served |
| Monitoring | https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/monitoring | quality, drift and alert status per adapter |
| Raw metrics | https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/metrics | everything Prometheus collects |

**MLflow shortcuts:**

| Page | Link |
|---|---|
| All registered models | https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com/#/models |
| SQL adapter (versions + aliases) | https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com/#/models/adaptos-adapter-sql |
| Experiments (data, training, evaluation, router, retraining) | https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com/#/experiments |

**Prometheus shortcuts:**

| Page | Link |
|---|---|
| Alerts | https://67j6ijkgl5.execute-api.us-east-1.amazonaws.com/alerts |
| Alert rules | https://67j6ijkgl5.execute-api.us-east-1.amazonaws.com/rules |
| Scrape targets | https://67j6ijkgl5.execute-api.us-east-1.amazonaws.com/targets |

---

## How to use the Gateway page

Open https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/docs, then:

1. Click an endpoint, for example `POST /chat`.
2. Click **Try it out**.
3. **Delete everything in the box** and paste in the text from the steps below.
4. Click **Execute**.
5. Scroll down to **Response body** to see the answer.

**Important:** don't leave `"adapter": "string"` in the box. That gives an error. Always replace the whole box.

---

## Step 1: Show that it's running

Open https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/models. You'll see `"sql": "4"`.

Say: *"These are the live versions of each model. SQL is on version 4, which the system created by retraining itself."*

## Step 2: Show the router picking the right hat

On the [Gateway page](https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/docs), use **`POST /chat`** four times, pasting each of these in turn.

**SQL question:**

```json
{"message": "CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INTEGER)\n\nQuestion: Which employees in the Sales department earn more than 50000?"}
```

**Summary:**

```json
{"message": "Summarize this conversation:\n\n#Person1#: Did you finish the report?\n#Person2#: Not yet, the data came in late.\n#Person1#: Please send me a draft by Friday."}
```

**Math:**

```json
{"message": "A bakery sells 12 muffins per tray. They bake 7 trays and sell all but 9 muffins. How many muffins did they sell?"}
```

**General question, answered with no hat:**

```json
{"message": "What is the capital of Japan?"}
```

In each response, point at:

- `adapter` and `adapter_version`: which hat answered, and which version.
- `route`: what the router decided, and how sure it was (`confidence`).
- `quality`: the automatic check of the answer. The general question shows `null` because it has no task check.

## Step 3: Feedback

From the SQL answer in Step 2, **copy the `request_id`**. Then use **`POST /feedback`**:

```json
{"request_id": "PASTE-THE-ID-HERE", "rating": "up"}
```

Say: *"Users can rate answers. Ratings override the automatic score, and corrections become new training data."*

## Step 4: Versions, hot-swap and the self-fix (the best part)

**4a.** Use **`POST /chat`** with a "new kind" of question:

```json
{"message": "CREATE TABLE table_name_22 (year INTEGER, wins VARCHAR, points VARCHAR)\n\nQuestion: What is the earliest year associated with 0 wins and 42 points?"}
```

The answer comes from **sql v4**: `SELECT MIN(year) ...` with **quality 1.0**. It correctly used `MIN`.

**4b.** Switch back to the old version with **`POST /admin/rollback`**:

- In the **x-admin-token** box, type `dev-admin-token`.
- In the body, paste:

```json
{"model": "sql"}
```

The response shows `"from": "4", "to": "1"`.

Say: *"I just swapped the model version live, without restarting anything."*

**4c.** Ask the **same question as 4a** again. Now **sql v1** answers `SELECT year FROM ...` with **quality 0.4**. It missed the `MIN`, because v1 never learned questions like this.

Say: *"This is the original version. When people started asking this kind of question, the monitor noticed the quality drop and raised an alarm. The system then retrained, tested and deployed v4 by itself."*

**4d.** Switch back to v4: **`POST /admin/rollback`** again with the same token and body. It shows `"from": "1", "to": "4"`. The system always keeps the previous version, so switching back takes one click.

Check it on https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/models: `"sql": "4"` again.

## Step 5: Monitoring

Open https://1axukpp3ah.execute-api.us-east-1.amazonaws.com/monitoring. For each hat it shows `quality`, `drift` (whether the questions are changing), `status` (ok or alert) and the thresholds.

Say: *"It watches the last 50 answers of each hat, and an alert triggers retraining."*

## Step 6: MLflow

- [Models](https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com/#/models) → [adaptos-adapter-sql](https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com/#/models/adaptos-adapter-sql): all versions, plus the **champion** and **previous** labels.
- [Experiments](https://b1rrv0apc8.execute-api.us-east-1.amazonaws.com/#/experiments):
  - **`adaptos-retraining`**: open the run marked **promoted**. It shows why it started (the alarm reason), the before and after scores (0.18 → 0.47 on the new questions), and the new version.
  - **`adaptos-training`**: a training run with its loss graph.
  - **`adaptos-evaluation`**: test scores, and each answer the model gave.
  - **`adaptos-data`**: the dataset versions and how many rows were cleaned out.

## Step 7: Grafana dashboard

Open https://iafg8o1nzf.execute-api.us-east-1.amazonaws.com/d/adaptos-overview. No login needed.

Show the green or red status per hat, the live versions, and the quality and drift graphs with dashed threshold lines. Any requests you just sent show up in the traffic graphs.

## Step 8 (optional): Alarm rules

- Alert rules: https://67j6ijkgl5.execute-api.us-east-1.amazonaws.com/rules
- Current alerts: https://67j6ijkgl5.execute-api.us-east-1.amazonaws.com/alerts

---

## Good to know

- **30-second limit.** The public links cut off any request after 30 seconds. Normal answers take 2–7 seconds. If a long math answer fails, ask again or add `"max_new_tokens": 100` to the body.
- **First request after idle** can take a few extra seconds.
- **The same trained state** as the development laptop: all model versions, MLflow history and the request log.
- **Self-healing on the GPU** is installed but not run. To run it, SSH into the server and run `~/retrain_once.sh`; it retrains on the T4 GPU in about 7 minutes.
- **Cost guardrail.** The server deletes itself 8 hours after launch (maximum about $4.30). Everything lives in one CloudFormation stack, `adaptos-demo`, and deleting that stack removes it all.
