"""
================================================================================
MAIN APP — PIPELINE ORCHESTRATOR + GRADIO UI
================================================================================
File    : app.py
Purpose : The main entry point for the AI Research Digest application.
          This file does two things:

          1. PIPELINE ORCHESTRATOR
             Wires together all 4 phases into a single run_pipeline() function:
             Phase 1 → Fetch papers (arXiv + HuggingFace)
             Phase 2 → Filter & rank (top 5 papers)
             Phase 3 → Summarise (Mistral 7B via HF Inference API)
             Phase 4 → Send newsletter (HTML email via Resend)

          2. GRADIO UI
             Builds the user-facing web interface where users can:
             - Enter their email address
             - Choose: run once now OR subscribe to daily digest
             - Click submit to trigger the pipeline

Deployment:
    - This file is what HuggingFace Spaces looks for by default
    - HF Spaces runs: python app.py
    - Gradio automatically creates a public URL for the Space

Environment Variables (set in HF Space Secrets):
    HF_TOKEN       : HuggingFace API token for Mistral 7B inference
    RESEND_API_KEY : Resend API key for email delivery
    SENDER_EMAIL   : Verified sender email (from your Resend domain)

Author  : AI Research Digest Project
================================================================================
"""

import os
import time
import atexit
import gradio as gr
from dotenv import load_dotenv

# ── Import all pipeline agents ────────────────────────────────────────────────
from agents.fetcher_arxiv   import ArxivFetcherAgent
from agents.fetcher_hf      import HuggingFaceFetcherAgent
from agents.filter_agent    import FilterRankAgent
from agents.summariser_agent import SummariserAgent
from agents.newsletter_agent import NewsletterAgent
from scheduler.job_scheduler import DigestScheduler

# Load .env for local development
# On HuggingFace Spaces, secrets are injected automatically
load_dotenv()


# ── Pipeline Configuration ────────────────────────────────────────────────────
# Centralise all tuneable settings here so they're easy to find and modify

CONFIG = {
    # How many papers to fetch from each source per run
    "arxiv_max_results" : 20,

    # How many hours back to look for papers
    "arxiv_hours_back"  : 48,   # 48hrs catches weekends + any missed days

    # How many papers to include in each newsletter
    "top_n_papers"      : 5,

    # HuggingFace papers to fetch
    "hf_max_results"    : 20,

    # Sender email (override with SENDER_EMAIL env var in production)
    # Resend's sandbox address works for testing without domain verification
    "sender_email"      : os.getenv("SENDER_EMAIL", "onboarding@resend.dev"),
    "sender_name"       : "AI Research Digest",

    # Daily schedule time (UTC)
    "schedule_hour"     : 8,    # 08:00 UTC = good morning time globally
    "schedule_minute"   : 0,
}


# ── Phase orchestration ───────────────────────────────────────────────────────

def run_pipeline(recipient_email: str, progress_callback=None):
    """
    Runs the full 4-phase pipeline for a given recipient email.

    progress_callback is an optional function(message: str) that gets called
    after each phase completes — used by handle_submit() to stream live
    status updates to the Gradio UI so the user sees progress in real time.

    Args:
        recipient_email   : The email address to deliver the newsletter to.
        progress_callback : Optional callable(str) for streaming UI updates.

    Returns:
        dict: Pipeline result containing success, message, papers_found,
              papers_sent, html_preview, email_id.
    """

    def notify(msg: str):
        """Sends a progress update to the UI if callback is registered."""
        print(msg)
        if progress_callback:
            progress_callback(msg)

    notify(f"🚀 Pipeline started for {recipient_email}\n")

    # ── Phase 1: Fetch ────────────────────────────────────────────────────────
    notify("📡 Phase 1 of 4 — Fetching papers...\n   Querying arXiv API...")

    arxiv_agent = ArxivFetcherAgent(
        max_results = CONFIG["arxiv_max_results"],
        hours_back  = CONFIG["arxiv_hours_back"],
    )
    arxiv_papers = arxiv_agent.fetch()
    notify(f"   ✅ arXiv: {len(arxiv_papers)} papers fetched\n   Scraping HuggingFace Papers...")

    hf_agent     = HuggingFaceFetcherAgent(max_results=CONFIG["hf_max_results"])
    hf_papers    = hf_agent.fetch()
    total_fetched = len(arxiv_papers) + len(hf_papers)

    notify(f"   ✅ HuggingFace: {len(hf_papers)} papers fetched\n   Total: {total_fetched} papers collected")

    if total_fetched == 0:
        return {
            "success": False,
            "message": "Could not fetch any papers. Please check your internet connection and try again.",
            "papers_found": 0, "papers_sent": 0, "html_preview": "", "email_id": None,
        }

    # ── Phase 2: Filter & Rank ────────────────────────────────────────────────
    notify(f"\n🔍 Phase 2 of 4 — Filtering & ranking {total_fetched} papers...")

    filter_agent = FilterRankAgent(top_n=CONFIG["top_n_papers"])
    top_papers   = filter_agent.run(arxiv_papers, hf_papers)

    notify(f"   ✅ Deduplicated, scored and ranked all papers\n   📰 Top {len(top_papers)} selected for newsletter:")
    for i, p in enumerate(top_papers, 1):
        short_title = p.title[:55] + "..." if len(p.title) > 55 else p.title
        notify(f"      [{i}] {short_title}")

    if not top_papers:
        return {
            "success": False,
            "message": "No papers passed the quality filter. Please try again later.",
            "papers_found": total_fetched, "papers_sent": 0, "html_preview": "", "email_id": None,
        }

    # ── Phase 3: Summarise ────────────────────────────────────────────────────
    notify(f"\n🤖 Phase 3 of 4 — Summarising with Mistral 7B...")
    notify(f"   This is the slowest step (~15 sec per paper × {len(top_papers)} papers)")

    summariser        = SummariserAgent()
    summarised_papers = []

    for i, paper in enumerate(top_papers, 1):
        short_title = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
        notify(f"   ⏳ Summarising paper {i}/{len(top_papers)}: {short_title}")
        result_single = summariser.run([paper])   # summarise one at a time for live updates
        summarised_papers.extend(result_single)
        notify(f"   ✅ Paper {i}/{len(top_papers)} done")

    notify(f"   ✅ All {len(summarised_papers)} summaries generated")

    # ── Phase 4: Send Newsletter ──────────────────────────────────────────────
    notify(f"\n📧 Phase 4 of 4 — Rendering & sending newsletter to {recipient_email}...")

    newsletter_agent = NewsletterAgent(
        sender_email = CONFIG["sender_email"],
        sender_name  = CONFIG["sender_name"],
    )
    result = newsletter_agent.run(summarised_papers, recipient_email)

    result["papers_found"] = total_fetched
    result["papers_sent"]  = len(summarised_papers)

    if result["success"]:
        notify(f"   ✅ Email sent! ID: {result.get('email_id', 'N/A')}")
    else:
        notify(f"   ❌ Email failed: {result['message']}")

    return result


def handle_submit(email: str, mode: str):
    """
    Called when the user clicks the Submit button in the Gradio UI.

    Uses a generator (yield) to stream live status updates to the UI
    after each pipeline phase — so the user sees real-time progress
    instead of a blank screen for 60-90 seconds.

    Each yield pushes an updated status string to the Gradio Textbox
    immediately, without waiting for the full pipeline to complete.

    Args:
        email : Email address entered by the user.
        mode  : "Send Now" or "Daily Schedule" from the Radio widget.

    Yields:
        tuple[str, str]: (status_message, html_preview)
    """
    # ── Input validation ──────────────────────────────────────────────────────
    email = email.strip() if email else ""

    if not email:
        yield "⚠️ Please enter your email address.", ""
        return

    if not _is_valid_email(email):
        yield "⚠️ Please enter a valid email address (e.g. you@example.com).", ""
        return

    # ── Mode: Send Now ────────────────────────────────────────────────────────
    if mode == "Send Now":

        # status_lines accumulates all progress messages into one string.
        # Each yield replaces the full textbox content — so we keep appending
        # new lines to show a running log of what has happened so far.
        status_lines = []

        def update_status(msg: str):
            """Appends a new line to the status log."""
            status_lines.append(msg)

        # Show initial message immediately so user knows it started
        status_lines.append(f"⏳ Starting pipeline for {email}...")
        status_lines.append("━" * 45)
        yield "\n".join(status_lines), ""

        try:
            # Run pipeline — progress_callback fires after each step
            # Each callback call appends to status_lines but we can't yield
            # from inside a callback, so we run phases manually with yields

            # ── Phase 1 ───────────────────────────────────────────────────────
            status_lines.append("\n📡 Phase 1 of 4 — Fetching papers...")
            status_lines.append("   Querying arXiv API...")
            yield "\n".join(status_lines), ""

            arxiv_agent  = ArxivFetcherAgent(
                max_results=CONFIG["arxiv_max_results"],
                hours_back =CONFIG["arxiv_hours_back"],
            )
            arxiv_papers = arxiv_agent.fetch()
            status_lines.append(f"   ✅ arXiv: {len(arxiv_papers)} papers fetched")
            status_lines.append("   Scraping HuggingFace Papers...")
            yield "\n".join(status_lines), ""

            hf_agent     = HuggingFaceFetcherAgent(max_results=CONFIG["hf_max_results"])
            hf_papers    = hf_agent.fetch()
            total_fetched = len(arxiv_papers) + len(hf_papers)
            status_lines.append(f"   ✅ HuggingFace: {len(hf_papers)} papers fetched")
            status_lines.append(f"   📊 Total collected: {total_fetched} papers")
            yield "\n".join(status_lines), ""

            if total_fetched == 0:
                status_lines.append("\n❌ No papers fetched — check internet connection.")
                yield "\n".join(status_lines), ""
                return

            # ── Phase 2 ───────────────────────────────────────────────────────
            status_lines.append(f"\n🔍 Phase 2 of 4 — Filtering & ranking {total_fetched} papers...")
            yield "\n".join(status_lines), ""

            filter_agent = FilterRankAgent(top_n=CONFIG["top_n_papers"])
            top_papers   = filter_agent.run(arxiv_papers, hf_papers)

            status_lines.append(f"   ✅ Top {len(top_papers)} papers selected:")
            for i, p in enumerate(top_papers, 1):
                short = p.title[:52] + "..." if len(p.title) > 52 else p.title
                status_lines.append(f"      [{i}] {short}")
            yield "\n".join(status_lines), ""

            if not top_papers:
                status_lines.append("\n❌ No papers passed quality filter. Try again later.")
                yield "\n".join(status_lines), ""
                return

            # ── Phase 3 ───────────────────────────────────────────────────────
            status_lines.append(f"\n🤖 Phase 3 of 4 — Summarising with Mistral 7B...")
            status_lines.append(f"   (~15 sec per paper × {len(top_papers)} papers)")
            yield "\n".join(status_lines), ""

            summariser        = SummariserAgent()
            summarised_papers = []
            total_summarise_start = time.time()

            for i, paper in enumerate(top_papers, 1):
                short = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
                status_lines.append(f"   ⏳ [{i}/{len(top_papers)}] Summarising: {short}")
                yield "\n".join(status_lines), ""

                paper_start = time.time()
                single = summariser.run([paper])
                paper_elapsed = round(time.time() - paper_start, 1)
                summarised_papers.extend(single)
                status_lines.append(f"   ✅ [{i}/{len(top_papers)}] Done in {paper_elapsed}s")
                yield "\n".join(status_lines), ""

            total_summarise_elapsed = round(time.time() - total_summarise_start, 1)
            status_lines.append(f"   ⏱ Total summarisation time: {total_summarise_elapsed}s")
            yield "\n".join(status_lines), ""

            # ── Phase 4 ───────────────────────────────────────────────────────
            status_lines.append(f"\n📧 Phase 4 of 4 — Sending newsletter to {email}...")
            yield "\n".join(status_lines), ""

            newsletter_agent = NewsletterAgent(
                sender_email = CONFIG["sender_email"],
                sender_name  = CONFIG["sender_name"],
            )
            result = newsletter_agent.run(summarised_papers, email)
            result["papers_found"] = total_fetched
            result["papers_sent"]  = len(summarised_papers)

            # ── Final status ──────────────────────────────────────────────────
            status_lines.append("━" * 45)
            if result["success"]:
                status_lines.append(f"✅ Newsletter delivered successfully!")
                status_lines.append(f"📧 Sent to      : {email}")
                status_lines.append(f"📄 Papers found : {total_fetched}")
                status_lines.append(f"📰 Papers sent  : {result['papers_sent']}")
                status_lines.append(f"🔑 Email ID     : {result.get('email_id', 'N/A')}")
                yield "\n".join(status_lines), result.get("html_preview", "")
            else:
                status_lines.append(f"❌ Failed to send: {result['message']}")
                yield "\n".join(status_lines), ""

        except Exception as e:
            status_lines.append(f"\n❌ Unexpected error: {str(e)}")
            status_lines.append("Please check your API keys in Space Secrets and try again.")
            yield "\n".join(status_lines), ""

    # ── Mode: Daily Schedule ──────────────────────────────────────────────────
    elif mode == "Daily Schedule":
        if scheduler.is_subscribed(email):
            yield (
                f"ℹ️ {email} is already subscribed to daily digests.\n"
                f"You will receive your next digest tomorrow at 08:00 UTC."
            ), ""
            return

        result = scheduler.add_job(
            email  = email,
            hour   = CONFIG["schedule_hour"],
            minute = CONFIG["schedule_minute"],
        )

        if result["success"]:
            yield (
                f"✅ Daily digest scheduled!\n\n"
                f"📧 Email    : {email}\n"
                f"🕐 Time     : 08:00 UTC every day\n"
                f"📅 Next run : {result['next_run']}\n\n"
                f"💡 Note: The Space must be active for scheduled emails to send.\n"
                f"   For guaranteed delivery, consider upgrading to a persistent Space."
            ), ""
        else:
            yield f"❌ Failed to schedule: {result['message']}", ""


def handle_unsubscribe(email: str) -> str:
    """
    Called when user clicks the Unsubscribe button.
    Cancels the daily scheduled job for the given email.

    Args:
        email: Email address to unsubscribe.

    Returns:
        str: Status message shown in the UI.
    """
    email = email.strip() if email else ""

    if not email:
        return "⚠️ Please enter your email address to unsubscribe."

    result = scheduler.remove_job(email)

    if result["success"]:
        return f"✅ {email} has been unsubscribed from daily digests."
    else:
        return f"ℹ️ {email} was not found in scheduled digests."


def _is_valid_email(email: str) -> bool:
    """
    Basic email format validation using regex.
    Catches obvious typos without being overly strict.

    Args:
        email: Email string to validate.

    Returns:
        bool: True if format looks valid, False otherwise.
    """
    import re
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks UI.

    Layout overview:
    ┌─────────────────────────────────────────────────┐
    │  Header — title + description                   │
    ├─────────────────────────────────────────────────┤
    │  Email input                                    │
    │  Mode selector (Send Now / Daily Schedule)      │
    │  Submit button                                  │
    ├─────────────────────────────────────────────────┤
    │  Status output (result message)                 │
    │  Tabs: [ Email Preview ] [ How it works ]       │
    ├─────────────────────────────────────────────────┤
    │  Unsubscribe section (collapsible)              │
    └─────────────────────────────────────────────────┘

    Returns:
        gr.Blocks: The fully configured Gradio application.
    """

    # Custom CSS for a polished dark theme that matches our email design
    custom_css = """
        /* ── Global ──────────────────────────────────────────────────── */
        body, .gradio-container {
            background-color: #0f1117 !important;
            font-family: Georgia, serif !important;
        }

        /* ── Header ──────────────────────────────────────────────────── */
        .header-box {
            background: linear-gradient(135deg, #1a1d2e, #12151f);
            border: 1px solid #2a2d3e;
            border-top: 3px solid #6c63ff;
            border-radius: 12px;
            padding: 36px;
            text-align: center;
            margin-bottom: 8px;
        }

        .header-eyebrow {
            font-family: 'Courier New', monospace;
            font-size: 11px;
            letter-spacing: 4px;
            color: #6c63ff;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .header-title {
            font-size: 38px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .header-title .accent { color: #6c63ff; }

        .header-subtitle {
            font-size: 15px;
            color: #8b8fa8;
            font-style: italic;
            margin-bottom: 20px;
        }

        .header-badges {
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .badge {
            font-family: 'Courier New', monospace;
            font-size: 10px;
            letter-spacing: 2px;
            padding: 5px 12px;
            border-radius: 100px;
            border: 1px solid #2a2d3e;
            color: #9ca3af;
            background: rgba(255,255,255,0.03);
        }

        /* ── Form card ────────────────────────────────────────────────── */
        .form-card {
            background: #1a1d2e;
            border: 1px solid #252840;
            border-radius: 12px;
            padding: 28px 32px;
        }

        /* ── How it works section ─────────────────────────────────────── */
        .how-it-works {
            background: #13161f;
            border: 1px solid #1e2130;
            border-radius: 12px;
            padding: 24px;
            font-size: 14px;
            color: #9ca3af;
            line-height: 1.8;
        }

        .how-it-works h3 {
            color: #f3f4f6;
            font-size: 16px;
            margin-bottom: 16px;
            font-family: Georgia, serif;
        }

        .pipeline-step {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 12px;
        }

        .step-number {
            font-family: 'Courier New', monospace;
            font-size: 10px;
            color: #6c63ff;
            background: rgba(108,99,255,0.1);
            border: 1px solid rgba(108,99,255,0.2);
            border-radius: 100px;
            padding: 2px 8px;
            white-space: nowrap;
            margin-top: 2px;
        }

        /* ── Footer ──────────────────────────────────────────────────── */
        .footer-note {
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            color: #374151;
            letter-spacing: 1px;
            padding: 16px;
        }
    """

    with gr.Blocks(
        title = "AI Research Digest",
        theme = gr.themes.Base(
            primary_hue   = "violet",
            neutral_hue   = "slate",
            font          = gr.themes.GoogleFont("Source Serif 4"),
        ),
        css = custom_css,
    ) as app:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <div class="header-eyebrow">Open Source · Powered by Mistral 7B</div>
            <div class="header-title">
                AI Research <span class="accent">Digest</span>
            </div>
            <div class="header-subtitle">
                The latest AI breakthroughs from arXiv & HuggingFace — explained in plain English
            </div>
            <div class="header-badges">
                <span class="badge">📡 arXiv + HuggingFace</span>
                <span class="badge">🤖 Mistral 7B Summaries</span>
                <span class="badge">📧 Email Delivery</span>
                <span class="badge">⏱ Daily or One-Time</span>
            </div>
        </div>
        """)

        # ── Main form ─────────────────────────────────────────────────────────
        with gr.Group(elem_classes="form-card"):

            gr.Markdown("### Get Your AI Research Digest")

            email_input = gr.Textbox(
                label       = "Your Email Address",
                placeholder = "you@example.com",
                info        = "We'll send today's top AI papers, summarised in plain English.",
                max_lines   = 1,
            )

            mode_selector = gr.Radio(
                choices = ["Send Now", "Daily Schedule"],
                value   = "Send Now",
                label   = "Delivery Mode",
                info    = (
                    "Send Now: Fetch & email today's papers immediately (takes ~60-90 sec). "
                    "Daily Schedule: Receive your digest every morning at 08:00 UTC."
                ),
            )

            submit_btn = gr.Button(
                value   = "🚀 Send My Digest",
                variant = "primary",
                size    = "lg",
            )

        # ── Status output ─────────────────────────────────────────────────────
        status_output = gr.Textbox(
            label     = "Status",
            lines     = 6,
            max_lines = 10,
            interactive = False,
            placeholder = "Your digest status will appear here after clicking Send...",
        )

        # ── Tabs: Email preview + How it works ────────────────────────────────
        with gr.Tabs():

            with gr.TabItem("📧 Email Preview"):
                gr.Markdown(
                    "*After sending, a preview of the email will appear here.*"
                )
                html_preview = gr.HTML(
                    label = "Rendered Newsletter Preview",
                    value = "",
                )

            with gr.TabItem("ℹ️ How It Works"):
                gr.HTML("""
                <div class="how-it-works">
                    <h3>How the Pipeline Works</h3>
                    <div class="pipeline-step">
                        <span class="step-number">Phase 1</span>
                        <span><strong style="color:#e8e6e1">Fetch</strong> — The app queries the arXiv API and scrapes HuggingFace Papers to collect the latest AI research published in the last 48 hours.</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-number">Phase 2</span>
                        <span><strong style="color:#e8e6e1">Filter</strong> — Papers are deduplicated across sources, filtered for quality, and ranked by recency, keyword relevance, and community interest. The top 5 are selected.</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-number">Phase 3</span>
                        <span><strong style="color:#e8e6e1">Summarise</strong> — Each paper is sent to Mistral 7B Instruct (running on HuggingFace Inference API) which rewrites the abstract into plain English with a headline, explanation, and everyday analogy.</span>
                    </div>
                    <div class="pipeline-step">
                        <span class="step-number">Phase 4</span>
                        <span><strong style="color:#e8e6e1">Deliver</strong> — The summaries are rendered into a styled HTML email and sent to your inbox via the Resend email API.</span>
                    </div>
                    <br/>
                    <p style="color:#6b7280; font-size:13px;">
                        Built with Python · Mistral 7B · Gradio · APScheduler · Resend<br/>
                        Deployed on HuggingFace Spaces · 100% open source
                    </p>
                </div>
                """)

        # ── Unsubscribe section ───────────────────────────────────────────────
        with gr.Accordion("🔕 Unsubscribe from Daily Digest", open=False):
            gr.Markdown(
                "Enter your email below to cancel your daily scheduled digest."
            )
            with gr.Row():
                unsub_email = gr.Textbox(
                    label       = "Email to Unsubscribe",
                    placeholder = "you@example.com",
                    max_lines   = 1,
                    scale       = 4,
                )
                unsub_btn = gr.Button(
                    value   = "Unsubscribe",
                    variant = "secondary",
                    scale   = 1,
                )
            unsub_status = gr.Textbox(
                label       = "Unsubscribe Status",
                interactive = False,
                max_lines   = 2,
            )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-note">
            AI RESEARCH DIGEST · OPEN SOURCE · BUILT ON HUGGINGFACE SPACES
        </div>
        """)

        # ── Wire up events ────────────────────────────────────────────────────

        # Submit button → handle_submit → update status + html preview
        submit_btn.click(
            fn      = handle_submit,
            inputs  = [email_input, mode_selector],
            outputs = [status_output, html_preview],
        )

        # Unsubscribe button → handle_unsubscribe → update unsub status
        unsub_btn.click(
            fn      = handle_unsubscribe,
            inputs  = [unsub_email],
            outputs = [unsub_status],
        )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Launches the Gradio app.

    Local development:
        python app.py
        → Opens at http://localhost:7860

    HuggingFace Spaces:
        HF Spaces automatically runs app.py and serves it publicly.
        Set your secrets (HF_TOKEN, RESEND_API_KEY, SENDER_EMAIL)
        in Space Settings → Repository Secrets before deploying.
    """
    app = build_ui()
    app.launch(
        # share=True creates a temporary public URL (useful for demos)
        # Set to False for HF Spaces (it creates its own public URL)
        share          = False,

        # show_error=True displays full Python tracebacks in the UI
        # Useful for development, consider False for production
        show_error     = True,

        # server_name="0.0.0.0" is required for HF Spaces Docker container
        # It makes the app accessible from outside the container
        server_name    = "0.0.0.0",

        # HF Spaces uses port 7860 by default
        server_port    = 7860,
    )
