# 🧠 AI Research Digest

> Stay ahead of AI research — without reading the papers.

An autonomous AI pipeline that fetches the latest research papers from **arXiv** and **HuggingFace Papers**, ranks them by relevance, summarises them in plain English using **Llama 3.1 8B via Groq**, and delivers a beautiful HTML newsletter to any inbox — daily or on demand.

Built entirely with free, open-source tools. Deployed on **HuggingFace Spaces**.

---

## 📌 Table of Contents

1. [What It Does](#-what-it-does)
2. [How It Works — Full Workflow](#-how-it-works--full-workflow)
3. [Architecture Diagram](#-architecture)
4. [Tech Stack & Why Each Tool Was Chosen](#-tech-stack--why-each-tool-was-chosen)
5. [Phase 1 — Fetching Papers](#-phase-1--fetching-papers)
6. [Phase 2 — Filtering & Ranking Logic](#-phase-2--filtering--ranking-logic)
7. [Phase 3 — AI Summarisation](#-phase-3--ai-summarisation)
8. [Phase 4 — Newsletter Delivery](#-phase-4--newsletter-delivery)
9. [Project Structure](#-project-structure)
10. [Environment Variables](#-environment-variables)
11. [Local Development Setup](#-local-development-setup)
12. [Deploying to HuggingFace Spaces](#-deploying-to-huggingface-spaces)
13. [Known Limitations](#-known-limitations)
14. [Total Cost](#-total-cost)

---

## ✨ What It Does

Every time the pipeline runs, it:

1. **Fetches** up to 100 recent AI papers from arXiv and HuggingFace Papers
2. **Deduplicates** papers that appear in both sources
3. **Scores and ranks** all papers using a multi-factor relevance algorithm
4. **Selects the top 5** with guaranteed source diversity (mix of arXiv + HF)
5. **Summarises** each paper using Llama 3.1 8B into 4 structured sections:
   - 📰 **Headline** — one punchy sentence capturing the breakthrough
   - 📖 **What it does** — 2-3 plain sentences, zero jargon
   - 💡 **Why it matters** — real-world impact for non-technical readers
   - 🔍 **Analogy** — "Think of it like..." to make it click instantly
6. **Renders** a dark-themed HTML email using a Jinja2 template
7. **Delivers** the newsletter via SendGrid to any email address

Users can run it once on demand, or subscribe to a daily automated digest at 08:00 UTC.

---

## 🔄 How It Works — Full Workflow

```
User opens HuggingFace Space
        │
        ▼
Enters email + clicks "Send Now"
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                         PIPELINE                               │
│                                                                │
│  PHASE 1 — FETCH                                               │
│  ├── ArXiv API → queries 5 categories (cs.AI, cs.LG,          │
│  │               cs.CL, cs.CV, stat.ML) → up to 82 papers     │
│  └── HF Scraper → scrapes huggingface.co/papers → 20 papers   │
│                                                                │
│  PHASE 2 — FILTER & RANK                                       │
│  ├── Merge all papers into one list (~102 total)               │
│  ├── Deduplicate (arXiv version preferred over HF duplicate)   │
│  ├── Quality filter (remove too-short titles/abstracts)        │
│  ├── Score each paper (recency + keywords + length)            │
│  ├── Sort by score descending                                  │
│  └── Balanced select: guarantee at least 1 arXiv + 1 HF paper │
│                                                                │
│  PHASE 3 — SUMMARISE                                           │
│  └── For each of the top 5 papers:                             │
│      ├── Build structured prompt                               │
│      ├── Call Groq API (Llama 3.1 8B Instruct)                │
│      ├── Parse response into 4 labelled sections               │
│      └── Store as SummarisedPaper object                       │
│                                                                │
│  PHASE 4 — DELIVER                                             │
│  ├── Render Jinja2 HTML email template with paper data         │
│  ├── Send via SendGrid REST API over HTTPS                     │
│  └── Return HTML preview to Gradio UI                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
User receives newsletter in inbox + sees HTML preview in UI
```

The UI streams **live progress updates** after each phase so users see exactly what's happening in real time — not a blank screen for 60-90 seconds.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  GRADIO UI (HuggingFace Space)               │
│         [Email Input]   [Send Now / Daily Schedule]          │
│         [Live Status Log]   [HTML Email Preview Tab]         │
└─────────────────────────┬────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │       app.py          │
              │   run_pipeline()      │  Orchestrates all phases
              │   handle_submit()     │  Streams progress to UI
              └───────────┬───────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│  fetcher_arxiv  │               │   fetcher_hf    │  PHASE 1
│  arXiv REST API │               │  BS4 Scraper    │
└────────┬────────┘               └────────┬────────┘
         └──────────────┬──────────────────┘
                ┌───────▼────────┐
                │  filter_agent  │  PHASE 2
                │  Score & Rank  │
                └───────┬────────┘
                ┌───────▼────────┐
                │summariser_agent│  PHASE 3
                │   Groq API     │
                │ Llama 3.1 8B   │
                └───────┬────────┘
                ┌───────▼────────┐
                │newsletter_agent│  PHASE 4
                │ Jinja2 + HTML  │
                │  SendGrid API  │
                └───────┬────────┘
                        │
              ┌─────────▼──────────┐
              │   User's Inbox +   │
              │   Gradio Preview   │
              └────────────────────┘

         ┌──────────────────────┐
         │   job_scheduler.py   │  OPTIONAL
         │    APScheduler       │  Daily cron jobs
         │   Daily 08:00 UTC    │  per subscribed email
         └──────────────────────┘
```

---

## 🛠️ Tech Stack & Why Each Tool Was Chosen

| Component | Technology | Why This Was Chosen |
|---|---|---|
| **UI & App** | [Gradio](https://gradio.app) | Native HuggingFace Spaces support, zero config, built-in generator streaming for live progress updates |
| **App Hosting** | [HuggingFace Spaces](https://huggingface.co/spaces) | Free hosting with public URL, perfect for portfolio projects, direct git deploy |
| **LLM** | [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | Meta open-source model, excellent instruction following, produces clean structured output |
| **LLM Provider** | [Groq](https://groq.com) | Free tier (14,400 req/day), responses in 1-2 seconds on custom LPU hardware, reliable — HF Inference API was shut down in 2025 |
| **arXiv Papers** | [arXiv API](https://arxiv.org/help/api) | Official REST API, completely free, no auth needed, returns structured Atom XML with full abstracts |
| **HF Papers** | BeautifulSoup scraper | HuggingFace Papers are community-curated — if it appears there, the community already considered it notable |
| **Email** | [SendGrid](https://sendgrid.com) | Free (100/day), sends to any email, works over HTTPS (HF Spaces blocks SMTP port 587), proper DKIM/SPF support |
| **Email Domain** | Custom domain via Porkbun | Required for DMARC alignment — Gmail blocks third-party senders using @gmail.com as From address |
| **Templating** | [Jinja2](https://jinja.palletsprojects.com) | Industry-standard Python templating, keeps HTML completely separate from Python logic |
| **Scheduling** | [APScheduler](https://apscheduler.readthedocs.io) | Lightweight in-process scheduler, no Redis or database needed |
| **HTTP** | [requests](https://requests.readthedocs.io) | Used for arXiv API, HF scraping, Groq API, and SendGrid API calls |
| **HTML Parsing** | BeautifulSoup4 + lxml | BS4 parses HF Papers HTML; lxml is the fastest available parser backend |
| **Env Management** | python-dotenv | Loads `.env` locally; HF Spaces injects secrets automatically |

### Why NOT these alternatives

| Alternative | Why We Didn't Use It |
|---|---|
| **LangChain / LangGraph** | Adds complexity without benefit for a linear pipeline |
| **OpenAI GPT** | Paid API — wanted 100% free stack |
| **HF Inference API** | Permanently shut down in 2025, returns 410 Gone |
| **Gmail SMTP** | HuggingFace Spaces blocks outbound port 587 |
| **Resend** | Free tier only sends to your own email without domain verification |

---

## 📡 Phase 1 — Fetching Papers

### arXiv Fetcher (`agents/fetcher_arxiv.py`)

Queries the [arXiv API](https://export.arxiv.org/api/query) — a REST endpoint returning Atom XML.

**Categories queried:**

| Category | Covers |
|---|---|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CL` | Computation & Language (NLP) |
| `cs.CV` | Computer Vision |
| `stat.ML` | Statistical Machine Learning |

**How it works:**
1. Builds query: `cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR stat.ML`
2. Filters to papers submitted in the last **48 hours**
3. Requests up to 25 papers per category (deduped to ~82 unique)
4. Parses Atom XML — extracts title, abstract, authors, URL, date, categories
5. Returns a list of `Paper` dataclass objects

**Why 48 hours?** Papers appear in arXiv batches. 24 hours sometimes misses papers submitted late the previous day.

### HuggingFace Papers Fetcher (`agents/fetcher_hf.py`)

Scrapes [huggingface.co/papers](https://huggingface.co/papers) using BeautifulSoup.

**Two-stage fetch:**
1. **Stage 1** — Scrape the listing page for paper titles and HF URLs
2. **Stage 2** — Fetch each individual paper page to get the full abstract

**Why scrape?** HuggingFace has no public API for their curated papers list.

**Why HF Papers at all?** Papers here are manually submitted by the ML community — it's a quality signal on top of raw recency.

---

## 🔍 Phase 2 — Filtering & Ranking Logic

### Step 1: Merge
Combines arXiv (~82) and HF (~20) papers into one list.

### Step 2: Deduplicate
Many papers appear in both sources. Checks by:
- **Paper ID** — strips `hf_` prefix from HF IDs and compares to arXiv IDs
- **Normalised title** — lowercase, remove punctuation, check for matches

When a duplicate is found, **the arXiv version is kept** — it provides richer metadata and longer abstracts.

### Step 3: Quality Filter
Removes papers where:
- Title is fewer than 10 characters
- Abstract is fewer than 100 characters

### Step 4: Scoring Algorithm

Each paper is scored out of ~100 points across 4 factors:

**Factor 1 — Recency (max 30 points)**
```
Published today        → 30 points
Published yesterday    → 20 points
Published 2 days ago   → 10 points
Older                  →  0 points
```

**Factor 2 — Title Keywords (max 25 points)**

Scans the title for high-impact AI keywords. Each keyword match adds points:
```python
HIGH_IMPACT_KEYWORDS = [
    "large language model", "llm", "gpt", "transformer", "attention",
    "diffusion", "generative", "foundation model", "reinforcement learning",
    "rlhf", "fine-tuning", "instruction tuning", "alignment", "safety",
    "reasoning", "chain of thought", "agent", "autonomous", "multimodal",
    "vision language", "text to image", "neural", "deep learning",
    "bert", "embedding", "retrieval", "rag", "hallucination", "benchmark"
]
```

**Factor 3 — Abstract Keywords (max 20 points)**
Same keyword list applied to the abstract. Lower weight than title — less signal-dense.

**Factor 4 — Abstract Length (max 15 points)**
```
800+ characters   → 15 points   (very detailed)
600+ characters   → 10 points   (good)
300+ characters   →  5 points   (acceptable)
< 300 characters  →  0 points
```

### Step 5: Balanced Selection

Papers are sorted by score. Then `_balanced_select()` picks the final 5:

1. **Guarantee at least 1 arXiv paper** — takes the top-scoring arXiv paper
2. **Guarantee at least 1 HF paper** — takes the top-scoring HF paper  
3. **Fill remaining 3 slots** with highest-scoring papers from either source

This prevents all 5 papers coming from the same provider on any given day.

---

## 🤖 Phase 3 — AI Summarisation

### Model: Llama 3.1 8B Instruct via Groq

**Why Llama 3.1 8B?** Meta open-source model, fine-tuned for instructions, excellent at structured output, fast on Groq hardware.

**Why Groq?** Free (14,400 req/day), 1-2 second responses, always reliable. The previous provider (HF Inference API) was permanently shut down in 2025.

### The Prompt

```
You are a science writer summarising AI research for non-technical readers.

Paper: {title}
Abstract: {first 800 characters of abstract}

Reply using EXACTLY these 4 labels. Do not skip any.

HEADLINE: [one sentence capturing the key finding]
WHAT IT DOES: [2 sentences explaining what was built, no jargon]
WHY IT MATTERS: [1 sentence on real-world impact]
ANALOGY: [one sentence starting with: Think of it like]
```

### Response Parsing — Multi-Strategy

The parser handles all possible model output variations:

**Strategy 1 — Label splitting (primary):**
Finds each label position using case-insensitive regex, extracts content between consecutive labels. Handles uppercase, lowercase, bold markdown, numbered prefixes, and preamble text.

**Strategy 2 — Paragraph fallback:**
If fewer than 2 labels found, maps paragraphs positionally: para 1 → headline, 2 → what it does, 3 → why it matters, 4 → analogy.

**Retry logic:** 3 attempts per paper with 20s and 30s waits. Falls back to abstract excerpt if all retries fail — the pipeline never completely breaks.

---

## 📧 Phase 4 — Newsletter Delivery

### Template (Jinja2)

`templates/email_template.html` is rendered with:
- List of SummarisedPaper objects (looped in template)
- Date, paper count, estimated read time, time of day greeting

**Design:** Dark theme, gradient header, per-paper cards with source badges (purple = arXiv, teal = HuggingFace), analogy callout box.

### Sending (SendGrid REST API)

Posts to `https://api.sendgrid.com/v3/mail/send` with both HTML and plain-text versions. Returns HTTP 202 on success.

**Why a custom domain is required:** Gmail enforces DMARC alignment — it rejects emails where the `From:` domain doesn't match the sending server's authenticated domain. DNS CNAME records in your domain provider authorise SendGrid to send on your behalf.

---

## 📁 Project Structure

```
ai-research-digest/
│
├── app.py                      # Entry point — Gradio UI + pipeline orchestrator
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Excludes .env, __pycache__, preview_email.html
├── README.md                   # This file
│
├── agents/
│   ├── __init__.py             # Exports all agents + Paper, SummarisedPaper
│   ├── fetcher_arxiv.py        # Phase 1 — arXiv API (5 categories, 48hr window)
│   ├── fetcher_hf.py           # Phase 1 — HF Papers scraper (2-stage fetch)
│   ├── filter_agent.py         # Phase 2 — Merge, dedupe, score, rank, balance
│   ├── summariser_agent.py     # Phase 3 — Llama 3.1 8B via Groq API
│   └── newsletter_agent.py     # Phase 4 — Jinja2 render + SendGrid delivery
│
├── scheduler/
│   ├── __init__.py             # Exports DigestScheduler
│   └── job_scheduler.py        # APScheduler — daily 08:00 UTC jobs per email
│
└── templates/
    └── email_template.html     # Dark-themed Jinja2 HTML email template
```

---

## 🔑 Environment Variables

| Variable | Required | Purpose | Where to Get |
|---|---|---|---|
| `HF_TOKEN` | Yes | Fetch papers from HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free Read token |
| `GROQ_API_KEY` | Yes | Llama 3.1 8B summarisation | [console.groq.com](https://console.groq.com) — free, starts with `gsk_` |
| `SENDGRID_API_KEY` | Yes | Send newsletter emails | [sendgrid.com](https://sendgrid.com) → Settings → API Keys |
| `SENDER_EMAIL` | Yes | From address in sent emails | Must be verified in SendGrid Sender Authentication |

---

## 💻 Local Development Setup

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/yourusername/ai-research-digest
cd ai-research-digest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Fill in HF_TOKEN, GROQ_API_KEY, SENDGRID_API_KEY, SENDER_EMAIL

# 4. Run
python app.py
# Opens at http://localhost:7860
```

---

## ☁️ Deploying to HuggingFace Spaces

```bash
# 1. Create Space at huggingface.co/new-space
#    SDK: Gradio | Hardware: CPU Basic (free)

# 2. Add secrets in Space Settings → Repository Secrets

# 3. Push
git remote add space https://huggingface.co/spaces/yourusername/ai-research-digest
git push space main
```

**SendGrid domain setup** (required to send to any email):
1. Buy a domain at [Porkbun](https://porkbun.com) (~$1/year for `.xyz`)
2. SendGrid → Sender Authentication → Domain Authentication → add 3 CNAME records to your DNS
3. Verify in SendGrid, then set `SENDER_EMAIL = digest@yourdomain.xyz`

---

## ⚠️ Known Limitations

| Limitation | Cause | Workaround |
|---|---|---|
| Scheduled jobs lost on Space restart | APScheduler is in-memory only | Use GitHub Actions cron for persistent scheduling |
| Space sleeps after 15min inactivity | HF free tier | Upgrade Space hardware or use a keep-alive ping |
| 100 emails/day limit | SendGrid free tier | Upgrade SendGrid or use multiple accounts |

---

## 💰 Total Cost

| Item | Cost |
|---|---|
| HuggingFace Spaces hosting | Free |
| Groq API — Llama 3.1 8B summarisation | Free (14,400 req/day) |
| SendGrid — email delivery | Free (100 emails/day) |
| HuggingFace token — paper fetching | Free |
| Domain — for email DMARC compliance | ~$1/year |
| **Total** | **~$1/year** |

---

## 📄 License

MIT — free to use, modify, and deploy.

---

*Built as an open-source portfolio project demonstrating autonomous AI pipelines, multi-source data aggregation, LLM integration, and email delivery — using entirely free infrastructure.*
