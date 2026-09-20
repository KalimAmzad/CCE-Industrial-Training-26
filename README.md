# AI/LLM Application Builder — student workspace

**CCE Industrial Training 2026 · Grow with Data**

Everything you build in this course lives here. One project, **ShopWise** (an e-commerce
customer-support assistant), grown across 16 sessions.

Sessions are published here as they are delivered, so expect this repo to grow week by week.

## Setup (once — about 15 minutes)

You install four things. **Python is not one of them.** `uv` downloads the exact version this course
runs on; installing Python yourself is the most common way to end up with two of them and a kernel
that fails in week 4.

| | What | Why | Get it |
|---|---|---|---|
| 1 | **Cursor** or **VS Code** | where you open and run the notebooks | [cursor.com](https://cursor.com) · [code.visualstudio.com](https://code.visualstudio.com) |
| 2 | **Git** | how you get this repo, and each new session as it lands | [git-scm.com/downloads](https://git-scm.com/downloads) |
| 3 | **uv** | builds the Python environment in one command | step 1 below |
| 4 | **Gemini API key** | free, no card | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

### 1. Install uv

**macOS / Linux** — open Terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** — open PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Then close that window and open a new one.** The installer edits your PATH and the window you ran
it in still has the old one. Confirm with `uv --version`.

### 2. Get the repo

This repo is private, so you must be signed in to the GitHub account you gave us. The editor does
the sign-in for you — no SSH keys, no tokens:

**Ctrl/Cmd+Shift+P** → type `Git: Clone` → paste the URL below → sign in when the browser opens →
pick a folder → **Open**.

```
https://github.com/KalimAmzad/CCE-Industrial-Training-26.git
```

New sessions appear here as they are taught. To collect them: **Source Control** panel → `⋯` →
**Pull**, or `git pull` in the terminal. Your own `.env` and your notebook edits are never touched
by a pull.

### 3. Add the two extensions

**Ctrl/Cmd+Shift+X** opens Extensions. Search for and install **Python** and **Jupyter** (both
published by Microsoft). Cursor and VS Code both need them; without Jupyter an `.ipynb` file opens
as unreadable JSON.

### 4. Build the environment

Open the built-in terminal — **Ctrl+`** — check the prompt is in the repo folder, and run:

```bash
uv sync
```

About a minute and roughly 1 GB. It reads `pyproject.toml` and `uv.lock`, downloads Python 3.12 if
your machine hasn't got it, creates `.venv/` and installs every package all 16 sessions need. There
is no per-session install: week 8's dependencies are already there, so no session ever opens with a
failed `pip install`, and everyone in the cohort is on identical versions.

### 5. Put your key in .env

```bash
cp .env.example .env            # macOS / Linux
Copy-Item .env.example .env     # Windows PowerShell
```

Open `.env` in the editor and paste your key after `GEMINI_API_KEY=` — no quotes, no spaces. That
one key is enough for sessions 1–3; session 4 adds a free LangSmith account and the file explains
every other line when you get there.

Keys are **yours**, created in your own accounts — we walk through every signup in session. Never
commit `.env`; it's gitignored, keep it that way.

> ⚠️ **Free-tier prompts are used to train Google's models.** Google's pricing page lists "content
> used to improve our products" as **Yes** for the free tier and **No** for paid. Never paste real
> customer data, real names, or anything belonging to an employer into a free-tier notebook. Every
> ticket in `data/tickets.yml` is invented for this reason.

### 6. Check it before class

```bash
cd session-01
uv run python check_setup.py
```

It prints PASS or FAIL for the Python version, the environment, every package, your `.env`, your key
and the shared data files — and tells you exactly what to do about each FAIL. It talks to no network
and spends no quota, so run it as often as you like. Fix the first FAIL and run it again; a later
one is often a symptom of an earlier one.

### 7. Open a notebook and pick the kernel

Open `session-01/session-01.ipynb`. Top right of the notebook: **Select Kernel** → **Python
Environments** → the one whose path ends in `CCE-Industrial-Training-26/.venv` (usually marked
*recommended*). Run a cell with **Shift+Enter**.

Prefer the browser? `uv run jupyter lab`, then **Kernel → Change Kernel** and choose the same
`.venv`.

### The kernel is the #1 cause of broken cells

`uv sync` installed everything into `.venv` at the repo root. Your notebook has to run **that**
Python — not a system one, and above all not Anaconda's. Every notebook's first cell prints the
interpreter it is using, so read that before debugging anything else.

| what you see | what it means | fix |
|---|---|---|
| `uv: command not found` | PATH from before the install | close the terminal, open a new one |
| `.ipynb` opens as JSON | Jupyter extension missing | step 3 |
| `ModuleNotFoundError: langchain` | wrong kernel | step 7 |
| `AttributeError: 'Client' object has no attribute 'interactions'` | wrong kernel — an older `google-genai` | step 7 |
| `DefaultCredentialsError: Your default credentials were not found` (session 4+) | wrong kernel — Anaconda ships LangChain 0.3, which reads only `GOOGLE_API_KEY` and never sees your `GEMINI_API_KEY` | step 7 |

That last one names neither LangChain nor the kernel, which is why session 4's setup cell now stops
you at the first cell and tells you which kernel to pick.

## How to work

Each session folder is self-contained:

```
session-05/
├── session-05.ipynb    # the lesson — follow along in class
├── exercises.ipynb     # do these after class (⭐ warm-up · ⭐⭐ apply · ⭐⭐⭐ stretch)
├── slides/index.html   # the session deck — open it in any browser, works offline
└── project/            # ShopWise as it stands at the end of this session — runnable
```

Worked **solutions are published after the exercise review** at the start of the following session,
so the exercises are worth attempting cold.

`project/` is a **snapshot**, not a shared codebase. `session-06/project/` is `session-05/project/`
plus that session's changes, so you can see exactly what moved:

```bash
diff -r session-05/project session-06/project
```

Missed a session or fell behind? Open the folder for the session you're on — it runs on its own. You
never need `git checkout` to catch up.

Notebooks are committed **with their outputs**, so you can read any session without a key or quota,
and compare what your run produced with what the instructor's did. Run every cell yourself anyway —
watching your own code produce the output teaches far more than reading it. If a cell misbehaves,
`Kernel → Restart and Run All` before debugging anything else.

## Session map

| # | Date (F11) | Session | Version | Keys you need |
|---|---|---|---|---|
| S01 | 2026-08-14 | Hello, AI | — setup | `GEMINI_API_KEY` (free) |
| S02 | 2026-08-15 | Talking to the machines | v0.1 | `GEMINI_API_KEY` |
| S03 | 2026-08-21 | From chat to software | v0.2 | `GEMINI_API_KEY` |
| S04 | 2026-08-22 | One interface to rule them all | v1 | + `LANGSMITH_*` (your own free account) |
| S05 | 2026-08-28 | Give it hands | v2 | same |
| S06 | 2026-08-29 | Remember and respond | v3 | same |
| S07 | 2026-09-04 | Safety rails | v4 | same |
| S08 | 2026-09-05 | It should know your policies | v5 | same — Qdrant runs **locally, no signup** |
| S09 | 2026-09-11 | Answer with receipts | v6 | same |
| S10 | 2026-09-12 | Open the black box | v7 | same |
| S11 | 2026-09-18 | Pause, resume, remember forever | v8 | same |
| S12 | 2026-09-19 | A team of specialists | v9 | same |
| S13 | 2026-09-25 | It needs an API | v10 | same |
| S14 | 2026-09-26 | A face for your assistant | v11 | same |
| S15 | 2026-10-02 | Ship it | v12 | + deploy account, Qdrant Cloud (free tier) |
| S16 | 2026-10-03 | Demo day | capstone | same |

Cost: the course runs on **Gemini's free tier** throughout, on `gemini-3.1-flash-lite` by default.
Staying inside the notebooks should cost you nothing.

Two capabilities are the exception, and the notebooks say so where they appear: **search grounding**
is free only on `gemini-2.5-flash`/`-flash-lite` (500 requests/day), and **image generation is not
free on any model**. Your instructor demos both from a billed key; those cells are written to explain
themselves rather than crash if you re-run them on a free key.

Google no longer publishes free-tier rate limits — check your own at
[ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits) and in
your AI Studio dashboard. We walk through both in session 1.

## Shared data

`data/` is one copy for the whole course — the ticket set, the customer images the model is shown,
and later the policy documents and order database. Session folders read from it; they never carry
their own copy.

| | What it is | First used |
|---|---|---|
| `data/tickets.yml` | 20 invented support tickets, hand-labelled with a ground-truth `category` | S01 |
| `data/prices.yml` | token prices per model, with a `verified` date | S03 |
| `data/images/` | photos a customer would attach to a ticket — e.g. headphones with a torn earcup pad | S01 §7.3 |
| `data/policies/` | ShopWise's returns, warranty and shipping documents | S08 |
