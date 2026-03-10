"""
================================================================================
PHASE 4 — NEWSLETTER AGENT
================================================================================
Module  : agents/newsletter_agent.py
Purpose : Takes the summarised papers from Phase 3, renders them into a
          polished HTML email using our Jinja2 template, and delivers the
          newsletter to the recipient via Gmail SMTP.

Two responsibilities in one module:
    1. RENDER  — Jinja2 fills our HTML template with real paper data
    2. DELIVER — Gmail SMTP sends the rendered HTML to the recipient's inbox

Why Gmail SMTP?
    - Completely free — no domain purchase needed
    - Sends to ANY email address (unlike Resend sandbox)
    - Uses Python's built-in smtplib — no extra SDK dependency
    - Just needs a Gmail account + App Password (2 min setup)
    - Reliable deliverability for portfolio/demo use

Gmail App Password setup (required — regular password won't work):
    1. Go to myaccount.google.com
    2. Security → 2-Step Verification → turn ON (required first)
    3. Security → 2-Step Verification → App passwords (at bottom)
    4. Select app: Mail → Select device: Other → type "AI Digest" → Generate
    5. Copy the 16-character password shown (e.g. abcd efgh ijkl mnop)
    6. Add to HF Space secrets as GMAIL_APP_PASSWORD (no spaces)

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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
from pathlib import Path

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
    it to the recipient using Gmail SMTP.

    Usage:
        agent = NewsletterAgent(sender_email="you@gmail.com")
        result = agent.run(summarised_papers, recipient_email="user@example.com")

    Environment Variables Required:
        GMAIL_ADDRESS      : Your full Gmail address (e.g. yourname@gmail.com)
        GMAIL_APP_PASSWORD : 16-character Gmail App Password (NOT your Gmail password)

    How to get Gmail App Password:
        1. Go to myaccount.google.com
        2. Security → 2-Step Verification → enable it (required)
        3. Security → 2-Step Verification → scroll to "App passwords"
        4. App: Mail | Device: Other (name it "AI Digest") → Generate
        5. Copy the 16-char password → add to .env / HF Secrets
    """

    # Gmail SMTP server settings — these never change for Gmail
    GMAIL_SMTP_HOST = "smtp.gmail.com"
    GMAIL_SMTP_PORT = 587          # Port 587 = TLS (more reliable than SSL/465)

    # Path to our Jinja2 HTML template
    TEMPLATE_DIR  = Path(__file__).parent.parent / "templates"
    TEMPLATE_FILE = "email_template.html"

    def __init__(self, sender_email: str = None, sender_name: str = "AI Research Digest"):
        """
        Initialises the newsletter agent and validates Gmail credentials.

        Args:
            sender_email : Gmail address to send from.
                           Falls back to GMAIL_ADDRESS env var if not provided.
            sender_name  : Display name shown in recipient's inbox.

        Raises:
            ValueError: If GMAIL_ADDRESS or GMAIL_APP_PASSWORD are not set.
        """
        # Load Gmail credentials from environment variables
        # Priority: constructor argument → environment variable
        self.gmail_address  = sender_email or os.getenv("GMAIL_ADDRESS")
        self.app_password   = os.getenv("GMAIL_APP_PASSWORD")
        self.sender_name    = sender_name

        # Validate both credentials are present before proceeding
        if not self.gmail_address:
            raise ValueError(
                "Gmail address not set.\n"
                "  → Add to .env: GMAIL_ADDRESS=yourname@gmail.com\n"
                "  → Or HF Secrets: GMAIL_ADDRESS = yourname@gmail.com"
            )

        if not self.app_password:
            raise ValueError(
                "Gmail App Password not set.\n"
                "  → Setup: myaccount.google.com → Security → 2-Step Verification → App passwords\n"
                "  → Add to .env: GMAIL_APP_PASSWORD=abcdefghijklmnop  (no spaces)\n"
                "  → Or HF Secrets: GMAIL_APP_PASSWORD = abcdefghijklmnop"
            )

        # Remove any spaces from app password (users sometimes copy with spaces)
        self.app_password = self.app_password.replace(" ", "")

        # Set up Jinja2 templating environment
        # FileSystemLoader tells Jinja2 where to find our .html templates
        # autoescape=True prevents XSS by escaping special HTML characters
        self.jinja_env = Environment(
            loader     = FileSystemLoader(str(self.TEMPLATE_DIR)),
            autoescape = select_autoescape(["html"]),
        )

        print(f"[Newsletter Agent] Initialised — sender: {self.gmail_address}")

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self, summarised_papers: list[SummarisedPaper],
            recipient_email: str) -> dict:
        """
        Full newsletter pipeline: render template → send email.

        Args:
            summarised_papers : Output from SummariserAgent — list of
                                SummarisedPaper objects ready to display.
            recipient_email   : Email address to deliver the newsletter to.

        Returns:
            dict: {
                "success"     (bool) : Whether the email was sent successfully
                "email_id"    (str)  : A generated reference ID for tracking
                "message"     (str)  : Human-readable status message
                "html_preview"(str)  : Rendered HTML (for Gradio preview tab)
            }
        """
        print(f"\n[Newsletter Agent] Starting newsletter pipeline...")
        print(f"[Newsletter Agent] Recipient : {recipient_email}")
        print(f"[Newsletter Agent] Papers    : {len(summarised_papers)}")

        # ── Step 1: Render HTML template ──────────────────────────────────────
        print(f"[Newsletter Agent] Step 1 — Rendering HTML template...")
        html_content = self._render_template(summarised_papers)
        print(f"[Newsletter Agent]   ✅ HTML rendered ({len(html_content):,} characters)")

        # ── Step 2: Send via Gmail SMTP ───────────────────────────────────────
        print(f"[Newsletter Agent] Step 2 — Sending via Gmail SMTP...")
        result = self._send_email(
            to_email     = recipient_email,
            html_content = html_content,
            paper_count  = len(summarised_papers),
        )

        if result["success"]:
            print(f"[Newsletter Agent] ✅ Email delivered! Ref: {result['email_id']}")
        else:
            print(f"[Newsletter Agent] ❌ Delivery failed: {result['message']}")

        # Include rendered HTML so Gradio UI can show a preview
        result["html_preview"] = html_content
        return result

    # ── Step 1: Template Rendering ────────────────────────────────────────────

    def _render_template(self, summarised_papers: list[SummarisedPaper]) -> str:
        """
        Renders the Jinja2 HTML email template with real paper data.

        Template variables injected (must match {{ variable }} in the HTML):
            papers         : List of SummarisedPaper objects (looped in template)
            date           : Today's date e.g. "Monday, March 9 2025"
            paper_count    : Number of papers in this edition
            read_time      : Estimated reading time in minutes
            time_of_day    : "morning" / "afternoon" / "evening"
            recipient_name : "Reader" (generic personalisation)

        Args:
            summarised_papers: List of SummarisedPaper objects from Phase 3

        Returns:
            str: Fully rendered HTML string ready to be sent as email body
        """
        now = datetime.now(timezone.utc)

        # Estimate reading time — ~200 words per paper at 238 words/minute
        estimated_words    = len(summarised_papers) * 200 + 100
        estimated_read_min = max(1, round(estimated_words / 238))

        context = {
            "papers"        : summarised_papers,
            "date"          : now.strftime("%A, %B %-d %Y"),
            "paper_count"   : len(summarised_papers),
            "read_time"     : estimated_read_min,
            "time_of_day"   : self._get_time_of_day(now.hour),
            "recipient_name": "Reader",
        }

        template     = self.jinja_env.get_template(self.TEMPLATE_FILE)
        html_content = template.render(**context)
        return html_content

    def _get_time_of_day(self, hour: int) -> str:
        """
        Returns a greeting word based on UTC hour.

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

    # ── Step 2: Gmail SMTP Delivery ───────────────────────────────────────────

    def _send_email(self, to_email: str, html_content: str,
                    paper_count: int) -> dict:
        """
        Sends the rendered HTML email via Gmail SMTP using TLS encryption.

        How Gmail SMTP works:
            1. Connect to smtp.gmail.com on port 587
            2. Start TLS encryption (starttls) — secures the connection
            3. Login with Gmail address + App Password
            4. Send the email as a MIME multipart message
            5. Quit the connection cleanly

        We use MIMEMultipart("alternative") which allows sending both
        a plain-text fallback AND the HTML version. Email clients that
        can't render HTML will show the plain text instead.

        Args:
            to_email     : Recipient email address
            html_content : Fully rendered HTML string from _render_template()
            paper_count  : Number of papers (used in subject line)

        Returns:
            dict: { "success": bool, "email_id": str, "message": str }
        """
        # Build subject line with today's date
        today_str = datetime.now(timezone.utc).strftime("%b %-d")
        subject   = f"🧠 AI Research Digest — {today_str} ({paper_count} papers)"

        try:
            # ── Build the MIME email message ──────────────────────────────────
            # MIMEMultipart("alternative") = email with both plain text + HTML
            # The email client picks the best version it can display
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"{self.sender_name} <{self.gmail_address}>"
            msg["To"]      = to_email

            # Plain text fallback — shown if recipient's email can't render HTML
            # Keep it simple — just direct them to check a proper email client
            plain_text = (
                f"AI Research Digest — {today_str}\n\n"
                f"Your digest contains {paper_count} AI research paper summaries.\n"
                f"Please view this email in an HTML-capable email client to see "
                f"the full formatted newsletter.\n\n"
                f"Powered by Mistral 7B and open-source tools."
            )

            # Attach both parts — plain text first, HTML second
            # RFC 2046: the LAST part is preferred by email clients
            # So HTML (attached second) will be shown when supported
            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # ── Connect and send via Gmail SMTP ───────────────────────────────
            # smtplib.SMTP() opens the connection
            # starttls() upgrades to encrypted TLS connection (required by Gmail)
            # login() authenticates with App Password
            # sendmail() delivers the message
            with smtplib.SMTP(self.GMAIL_SMTP_HOST, self.GMAIL_SMTP_PORT) as server:
                server.ehlo()           # Identify ourselves to the SMTP server
                server.starttls()       # Upgrade to TLS encrypted connection
                server.ehlo()           # Re-identify after TLS upgrade
                server.login(self.gmail_address, self.app_password)
                server.sendmail(
                    from_addr = self.gmail_address,
                    to_addrs  = [to_email],
                    msg       = msg.as_string(),
                )

            # Generate a simple reference ID for tracking
            # (Gmail SMTP doesn't return an ID like Resend does)
            email_id = f"gmail_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

            return {
                "success" : True,
                "email_id": email_id,
                "message" : f"Email successfully sent to {to_email} via Gmail SMTP",
            }

        except smtplib.SMTPAuthenticationError:
            # Wrong Gmail address or App Password
            # Most common error — always check App Password setup first
            return {
                "success" : False,
                "email_id": None,
                "message" : (
                    "Gmail authentication failed.\n"
                    "Check that GMAIL_ADDRESS and GMAIL_APP_PASSWORD are correct.\n"
                    "Make sure you're using an App Password, not your regular Gmail password.\n"
                    "Setup: myaccount.google.com → Security → 2-Step Verification → App passwords"
                ),
            }

        except smtplib.SMTPRecipientsRefused:
            # Invalid recipient email address
            return {
                "success" : False,
                "email_id": None,
                "message" : f"Recipient address rejected by Gmail: {to_email}",
            }

        except smtplib.SMTPException as e:
            # Other SMTP-level errors
            return {
                "success" : False,
                "email_id": None,
                "message" : f"SMTP error: {str(e)}",
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

        Args:
            summarised_papers : Papers to render into the preview
            output_path       : Where to save the HTML file

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
    Standalone test for the Newsletter Agent with Gmail SMTP.

    Requirements:
        GMAIL_ADDRESS      = yourname@gmail.com  (in .env)
        GMAIL_APP_PASSWORD = abcdefghijklmnop    (in .env)
        TEST_EMAIL         = your@email.com      (in .env)
    """
    from agents.fetcher_arxiv import Paper
    from agents.summariser_agent import SummarisedPaper

    mock_papers = [
        SummarisedPaper(
            paper=Paper(
                paper_id       = "test_001",
                title          = "Large Language Models for Reasoning",
                authors        = ["Alice Smith", "Bob Jones"],
                abstract       = "We explore chain-of-thought reasoning...",
                published_date = "2025-03-09",
                url            = "https://arxiv.org/abs/test_001",
                source         = "arxiv",
                categories     = ["cs.AI"],
            ),
            headline       = "AI models can now show their working — and it makes them smarter",
            what_it_does   = "Researchers found that asking AI to explain its reasoning step-by-step dramatically improves accuracy on complex tasks.",
            why_it_matters = "This means AI assistants become far more reliable for tasks requiring careful thinking.",
            analogy        = "Think of it like asking a student to show their working in a maths exam.",
            summary_raw    = "",
        ),
    ]

    print("="*60)
    print("  NEWSLETTER AGENT — GMAIL SMTP TEST")
    print("="*60)

    try:
        agent = NewsletterAgent()

        # Save preview HTML
        agent.save_preview(mock_papers, "preview_email.html")
        print("✅ HTML preview saved → open preview_email.html in browser")

        # Send test email if TEST_EMAIL is set
        test_email = os.getenv("TEST_EMAIL")
        if test_email:
            print(f"\n📧 Sending test email to: {test_email}")
            result = agent.run(mock_papers, test_email)
            if result["success"]:
                print(f"✅ Email sent! Ref: {result['email_id']}")
            else:
                print(f"❌ Failed: {result['message']}")
        else:
            print("\n💡 Add TEST_EMAIL=your@email.com to .env to send a test email")

    except ValueError as e:
        print(f"\n⚠️  Setup required:\n{e}")
