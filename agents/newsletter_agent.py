"""
================================================================================
PHASE 4 — NEWSLETTER AGENT
================================================================================
Module  : agents/newsletter_agent.py
Purpose : Takes the summarised papers from Phase 3, renders them into a
          polished HTML email using our Jinja2 template, and delivers the
          newsletter to the recipient via the Resend email API.

Two responsibilities in one module:
    1. RENDER  — Jinja2 fills our HTML template with real paper data
    2. DELIVER — Resend API sends the rendered HTML to the recipient's inbox

Why Resend?
    - Modern email API with a clean Python SDK
    - Generous free tier: 3,000 emails/month, 100/day
    - Reliable deliverability (doesn't land in spam like raw SMTP)
    - One function call to send: resend.Emails.send({...})
    - Docs: https://resend.com/docs

Why Jinja2 for templating?
    - Industry standard Python templating engine (used in Flask, Django)
    - Keeps HTML completely separate from Python logic
    - Supports loops ({% for paper in papers %}), conditionals, and filters
    - Easy to update the email design without touching Python code

Pipeline position:
    Phase 1 (Fetch) → Phase 2 (Filter) → Phase 3 (Summarise) → Phase 4 (HERE)

Author  : AI Research Digest Project
================================================================================
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import resend
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv

# Import our data models from previous phases
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.summariser_agent import SummarisedPaper

# Load .env for local development
# On HuggingFace Spaces, secrets are injected automatically via Space settings
load_dotenv()


# ── Newsletter Agent ──────────────────────────────────────────────────────────

class NewsletterAgent:
    """
    Renders summarised papers into a beautiful HTML email and delivers
    it to the recipient using the Resend email API.

    Usage:
        agent = NewsletterAgent(sender_email="digest@yourdomain.com")
        result = agent.run(summarised_papers, recipient_email="user@example.com")

    Environment Variables Required:
        RESEND_API_KEY : Your Resend API key.
                         Get one free at https://resend.com
                         → Create account → API Keys → Create Key
                         → Set in .env locally or HF Space secrets

    Setup notes:
        - Resend requires a verified sender domain for production use.
        - For testing/dev, Resend provides a free sandbox domain:
          onboarding@resend.dev (limited to your own verified email address)
        - For production: verify your domain at resend.com/domains
    """

    # Path to our Jinja2 HTML template
    # Using Path(__file__) ensures this works regardless of where Python is run from
    TEMPLATE_DIR  = Path(__file__).parent.parent / "templates"
    TEMPLATE_FILE = "email_template.html"

    def __init__(self, sender_email: str = "onboarding@resend.dev",
                 sender_name: str = "AI Research Digest"):
        """
        Initialises the newsletter agent.

        Args:
            sender_email : The "From" address in the email.
                           Must be a verified domain in your Resend account.
                           Default uses Resend's sandbox (good for dev/testing).
            sender_name  : Display name shown in the recipient's inbox.
                           e.g. "AI Research Digest" appears as sender.

        Raises:
            ValueError: If RESEND_API_KEY is not set in environment variables.
        """
        # Load and validate the Resend API key
        self.resend_api_key = os.getenv("RESEND_API_KEY")
        if not self.resend_api_key:
            raise ValueError(
                "RESEND_API_KEY environment variable is not set.\n"
                "  → Get a free key at: https://resend.com\n"
                "  → Add to .env file: RESEND_API_KEY=re_your_key_here\n"
                "  → Or add to HF Space secrets in Settings → Repository Secrets"
            )

        # Set the API key on the resend SDK (it uses a module-level config)
        resend.api_key = self.resend_api_key

        # Compose the full "From" field: "Name <email@domain.com>"
        self.sender = f"{sender_name} <{sender_email}>"

        # Set up Jinja2 templating environment
        # FileSystemLoader tells Jinja2 where to find our .html templates
        # autoescape=True escapes HTML characters in variables (XSS prevention)
        self.jinja_env = Environment(
            loader      = FileSystemLoader(str(self.TEMPLATE_DIR)),
            autoescape  = select_autoescape(["html"]),
        )

        print(f"[Newsletter Agent] Initialised — sender: {self.sender}")

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self, summarised_papers: list[SummarisedPaper],
            recipient_email: str) -> dict:
        """
        Full newsletter pipeline: render template → send email.

        Args:
            summarised_papers : Output from SummariserAgent.run() — list of
                                SummarisedPaper objects ready to be displayed.
            recipient_email   : The email address to deliver the newsletter to.
                                Comes from the user's input in the Gradio UI.

        Returns:
            dict: Result containing:
                  - "success" (bool)  : Whether the email was sent successfully
                  - "email_id" (str)  : Resend's unique ID for the sent email
                  - "message" (str)   : Human-readable status message
                  - "html_preview"(str): The rendered HTML (useful for debugging)

        Example:
            result = agent.run(papers, "user@example.com")
            if result["success"]:
                print(f"Email sent! ID: {result['email_id']}")
        """
        print(f"\n[Newsletter Agent] Starting newsletter pipeline...")
        print(f"[Newsletter Agent] Recipient : {recipient_email}")
        print(f"[Newsletter Agent] Papers    : {len(summarised_papers)}")

        # ── Step 1: Render the HTML template ─────────────────────────────────
        print(f"[Newsletter Agent] Step 1 — Rendering HTML template...")
        html_content = self._render_template(summarised_papers)
        print(f"[Newsletter Agent]   ✅ HTML rendered ({len(html_content):,} characters)")

        # ── Step 2: Send via Resend API ───────────────────────────────────────
        print(f"[Newsletter Agent] Step 2 — Sending email via Resend...")
        result = self._send_email(
            to_email     = recipient_email,
            html_content = html_content,
            paper_count  = len(summarised_papers),
        )

        if result["success"]:
            print(f"[Newsletter Agent] ✅ Email delivered! Resend ID: {result['email_id']}")
        else:
            print(f"[Newsletter Agent] ❌ Delivery failed: {result['message']}")

        # Include the rendered HTML in the result so Gradio UI can preview it
        result["html_preview"] = html_content
        return result

    # ── Step 1: Template Rendering ────────────────────────────────────────────

    def _render_template(self, summarised_papers: list[SummarisedPaper]) -> str:
        """
        Renders the Jinja2 HTML email template with real paper data.

        Template variables injected (must match {{ variable }} in the HTML):
            papers        : List of SummarisedPaper objects (iterated in template)
            date          : Today's date e.g. "Monday, March 9 2025"
            paper_count   : Number of papers in this edition
            read_time     : Estimated reading time in minutes
            time_of_day   : "morning" / "afternoon" / "evening" (greeting)
            recipient_name: "Reader" (personalisation placeholder)

        Args:
            summarised_papers: List of SummarisedPaper objects from Phase 3

        Returns:
            str: Fully rendered HTML string ready to be sent as email body
        """
        # Build the template context — all variables the template can access
        now = datetime.now(timezone.utc)

        # Estimate reading time: ~200 words per paper summary + header/footer
        # Average person reads ~238 words/minute
        estimated_words    = len(summarised_papers) * 200 + 100
        estimated_read_min = max(1, round(estimated_words / 238))

        context = {
            # The main data — list of SummarisedPaper objects
            # Jinja2 template iterates over this with {% for item in papers %}
            "papers"        : summarised_papers,

            # Formatted date for the email header
            # strftime codes: %A=Monday, %B=March, %-d=9, %Y=2025
            "date"          : now.strftime("%A, %B %-d %Y"),

            # Stats shown in the header badge
            "paper_count"   : len(summarised_papers),
            "read_time"     : estimated_read_min,

            # Personalised greeting based on UTC hour
            # (approximate — users are in different timezones)
            "time_of_day"   : self._get_time_of_day(now.hour),

            # Recipient name — using generic "Reader" for now
            # Future enhancement: personalise from user profile
            "recipient_name": "Reader",
        }

        # Load and render the template
        template     = self.jinja_env.get_template(self.TEMPLATE_FILE)
        html_content = template.render(**context)

        return html_content

    def _get_time_of_day(self, hour: int) -> str:
        """
        Returns a greeting word based on the hour of the day (UTC).

        Args:
            hour: UTC hour (0-23)

        Returns:
            str: "morning", "afternoon", or "evening"
        """
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        else:
            return "evening"

    # ── Step 2: Email Delivery ────────────────────────────────────────────────

    def _send_email(self, to_email: str, html_content: str,
                    paper_count: int) -> dict:
        """
        Sends the rendered HTML email via the Resend API.

        Resend API call structure:
            resend.Emails.send({
                "from"    : "Name <email@domain.com>",
                "to"      : ["recipient@example.com"],
                "subject" : "...",
                "html"    : "<html>...</html>",
            })

        The API returns an object with an "id" field on success,
        or raises an exception on failure.

        Args:
            to_email     : Recipient's email address
            html_content : Fully rendered HTML string from _render_template()
            paper_count  : Number of papers (used in subject line)

        Returns:
            dict: { "success": bool, "email_id": str, "message": str }
        """
        # Build the email subject line with today's date
        today_str = datetime.now(timezone.utc).strftime("%b %-d")
        subject   = f"🧠 AI Research Digest — {today_str} ({paper_count} papers)"

        try:
            # Send via Resend SDK
            # "to" accepts a list — supports multiple recipients in future
            response = resend.Emails.send({
                "from"   : self.sender,
                "to"     : [to_email],
                "subject": subject,
                "html"   : html_content,
            })

            # Resend returns an object with an id on success
            email_id = getattr(response, "id", None) or response.get("id", "unknown")

            return {
                "success" : True,
                "email_id": email_id,
                "message" : f"Email successfully sent to {to_email}",
            }

        except resend.exceptions.ResendError as e:
            # Resend-specific errors (invalid API key, unverified domain, etc.)
            return {
                "success" : False,
                "email_id": None,
                "message" : f"Resend API error: {str(e)}",
            }

        except Exception as e:
            # Catch-all for unexpected errors (network issues, etc.)
            return {
                "success" : False,
                "email_id": None,
                "message" : f"Unexpected error sending email: {str(e)}",
            }

    # ── Utility: HTML Preview ─────────────────────────────────────────────────

    def save_preview(self, summarised_papers: list[SummarisedPaper],
                     output_path: str = "preview_email.html") -> str:
        """
        Saves the rendered HTML to a local file for visual preview.
        Useful during development to check the email design without sending.

        Usage:
            agent.save_preview(summarised_papers, "preview.html")
            # Then open preview.html in your browser

        Args:
            summarised_papers : Papers to render into the preview
            output_path       : Where to save the HTML file (default: current dir)

        Returns:
            str: Absolute path to the saved preview file
        """
        html = self._render_template(summarised_papers)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        abs_path = os.path.abspath(output_path)
        print(f"[Newsletter Agent] 👁️  Preview saved: {abs_path}")
        return abs_path


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone test for the Newsletter Agent.

    This test does TWO things:
        1. Renders the HTML template with mock data → saves preview.html
           (you can open this in your browser to check the design)
        2. Optionally sends a real test email if RESEND_API_KEY is set

    To send a real test email:
        1. Sign up at https://resend.com (free)
        2. Copy your API key
        3. Add to .env: RESEND_API_KEY=re_your_key_here
        4. Run: python agents/newsletter_agent.py
    """
    from agents.fetcher_arxiv import Paper
    from agents.summariser_agent import SummarisedPaper

    # ── Create mock SummarisedPaper objects ───────────────────────────────────
    mock_papers = [
        SummarisedPaper(
            paper=Paper(
                paper_id       = "2401.00001",
                title          = "Scaling Reasoning in Large Language Models via Chain-of-Thought",
                authors        = ["Alice Smith", "Bob Jones", "Carol White"],
                abstract       = "We explore how chain-of-thought prompting improves reasoning...",
                published_date = "2025-03-09",
                url            = "https://arxiv.org/abs/2401.00001",
                source         = "arxiv",
                categories     = ["cs.AI"],
            ),
            headline       = "AI models can now 'show their working' — and it makes them dramatically smarter",
            what_it_does   = "Researchers discovered that asking AI to explain its reasoning step-by-step, rather than just giving an answer, dramatically improves accuracy on complex problems. This works especially well for maths, logic puzzles, and multi-step decisions.",
            why_it_matters = "This means AI assistants can become much more reliable for tasks that require careful thinking, like analysing contracts or diagnosing problems — not just answering simple questions.",
            analogy        = "Think of it like asking a student to show their working in a maths exam — the process of writing it out forces clearer thinking and catches mistakes before the final answer.",
            summary_raw    = "Mock summary",
        ),
        SummarisedPaper(
            paper=Paper(
                paper_id       = "hf_2401.00002",
                title          = "Autonomous Coding Agents: From Bug Fixing to Full Feature Development",
                authors        = ["Dan Lee", "Eva Brown"],
                abstract       = "We present an autonomous agent that can write, test and debug code...",
                published_date = "2025-03-09",
                url            = "https://huggingface.co/papers/2401.00002",
                source         = "huggingface",
                categories     = ["cs.AI"],
            ),
            headline       = "AI can now write, test, and fix its own code — without any human help",
            what_it_does   = "A new AI system was built that takes a plain-English description of a software feature and produces working, tested code entirely on its own. It can even find bugs, rewrite the broken parts, and verify the fix — all in one go.",
            why_it_matters = "Software development costs could fall dramatically as AI takes on routine coding tasks, freeing human developers to focus on design and strategy rather than repetitive implementation work.",
            analogy        = "Think of it like hiring a junior developer who never sleeps, never gets frustrated, and automatically re-reads the manual whenever something doesn't work.",
            summary_raw    = "Mock summary",
        ),
        SummarisedPaper(
            paper=Paper(
                paper_id       = "2401.00003",
                title          = "Safety Alignment in Foundation Models: A New Benchmark",
                authors        = ["Frank Zhang", "Grace Kim", "Henry Park"],
                abstract       = "We introduce a comprehensive benchmark for evaluating AI safety...",
                published_date = "2025-03-08",
                url            = "https://arxiv.org/abs/2401.00003",
                source         = "arxiv",
                categories     = ["cs.AI"],
            ),
            headline       = "Researchers build the first proper report card for measuring AI safety",
            what_it_does   = "A team created a standardised test suite that measures how well AI systems follow safety guidelines across hundreds of real-world scenarios. It covers everything from refusing harmful requests to being honest about uncertainty.",
            why_it_matters = "Without consistent ways to measure safety, companies can't prove their AI is trustworthy. This benchmark gives regulators, businesses, and researchers a shared language for evaluating AI risk.",
            analogy        = "Think of it like crash-test safety ratings for cars — before these existed, there was no reliable way to compare how safe different vehicles were.",
            summary_raw    = "Mock summary",
        ),
    ]

    print("="*60)
    print("  NEWSLETTER AGENT — TEST RUN")
    print("="*60)

    try:
        agent = NewsletterAgent()

        # Always save a local HTML preview
        preview_path = agent.save_preview(mock_papers, "preview_email.html")
        print(f"\n✅ HTML preview saved → open in browser: {preview_path}")

        # Optionally send a real email (requires RESEND_API_KEY in .env)
        test_recipient = os.getenv("TEST_EMAIL")
        if test_recipient:
            print(f"\n📧 Sending test email to: {test_recipient}")
            result = agent.run(mock_papers, test_recipient)
            if result["success"]:
                print(f"✅ Email sent! ID: {result['email_id']}")
            else:
                print(f"❌ Failed: {result['message']}")
        else:
            print("\n💡 To test real email sending:")
            print("   Add TEST_EMAIL=your@email.com to your .env file")

    except ValueError as e:
        print(f"\n⚠️  Setup required:\n{e}")
