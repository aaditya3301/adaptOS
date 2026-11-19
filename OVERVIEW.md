# AdaptOS: Simple Overview

## What is it?

AdaptOS is a **smart AI assistant that can fix itself**.

It is one AI model that is good at three jobs:

- **SQL**: turns a question into a database query
- **Summarize**: turns a long chat into a short summary
- **Math**: solves word problems step by step

For anything else, like "What is the capital of Japan?", it answers as a normal assistant.

## The big idea

Think of the AI model as **one brain** that can wear different **hats**.

- The brain is a general AI model. It is OK at everything, but not great at anything.
- Each hat is a small add-on, called an **adapter**, that makes the brain an expert at one job.
- There is a SQL hat, a summary hat and a math hat.

The hats are tiny, so one brain can carry all of them. That is much cheaper than running three separate AI models.

## What it does, step by step

1. **A question comes in.**
2. **The switchboard picks the right hat.** A small helper, called the **router**, reads the question and decides: is this SQL, a summary, math, or a general question?
3. **The AI answers** wearing that hat. For general questions it uses no hat.
4. **Every answer gets checked.** The system gives each answer a quality score. For example, it checks that a SQL query actually works.
5. **Everything is watched.** The system keeps an eye on the last 50 answers of each hat.
6. **If a hat starts doing badly, the system raises an alarm.** This usually happens when people start asking a new kind of question it never learned.
7. **It fixes itself.** It teaches the hat the new kind of question, tests the improved hat, and switches to it only if it is really better. All of this happens without stopping the service.

## A real example from our project

- **The problem.** The SQL hat had learned simple questions, like "Which players are in team X?". It had never seen questions like **"How many…?"** or **"What is the total…?"**.
- **What we did.** We started sending it those new questions.
- **What happened next:**
  1. Its answers got worse, and the system **raised an alarm**.
  2. The system **retrained the SQL hat** with examples of the new questions.
  3. It **tested** the new hat against the old one.
  4. The new hat was better at both old and new questions, so the system **switched to it automatically**.
  5. The alarm went away.

| | Before fixing itself | After fixing itself |
|---|---|---|
| Correct answers on normal questions | 70% | **80%** |
| Correct answers on the new "How many…?" questions | 18% | **47%** |
| Status | Alarm | **Healthy** |

## Why we built it (the purpose)

AI models get worse over time. That is not because they break; it is because **the questions people ask change**. Normally a person has to notice this, retrain the model, test it and put it back online by hand.

This project shows how to **automate that whole cycle**. This is called **MLOps**, short for "Machine Learning Operations": running AI models reliably, the way good software teams run normal software.

The full cycle we built:

**Train → Save versions → Test → Put online → Watch → Retrain → (repeat)**

## The main parts

| Part | What it does, in simple words |
|---|---|
| **Data** | Collects and cleans the examples the AI learns from, and gives every dataset a version number |
| **Training** | Teaches each hat its job |
| **MLflow** (model library) | Stores every version of every hat with its scores, like a library with a history |
| **Testing + gate** | Checks a new hat on a fixed set of questions; it goes online only if it beats the current one |
| **Router** (switchboard) | Picks the right hat for each question |
| **Gateway** (the service) | The app people send questions to; it can swap hats without restarting |
| **Monitoring** | Watches answer quality and notices when questions start to look different |
| **Dashboard** (Grafana) | Graphs and alarms you can look at |
| **Controller** (the fixer) | Sees an alarm and runs the whole "retrain, test, switch" process by itself |

## Safety rules

- A new hat goes online **only if it passes the tests**.
- The previous hat is **always kept**, so we can **switch back with one click** if needed.
- The fixer cannot retrain the same hat over and over; there is a **waiting time and a daily limit**.
- Everything is **recorded**: which version is online, why it was retrained, and how it scored.

## How to see it working

With the project running:

| Open this | To see |
|---|---|
| http://localhost:8000/docs | Ask the AI questions |
| http://localhost:5000 | All model versions and their test scores (MLflow) |
| http://localhost:3000 | The live dashboard with graphs and alarms (Grafana) |

To watch the whole story happen live (question → alarm → self-fix → healthy), run:

```bash
python scripts/demo.py
```

## In one sentence

**AdaptOS is one AI that wears different expert "hats", picks the right hat for each question, notices when a hat starts doing badly, and teaches, tests and replaces that hat by itself.**
