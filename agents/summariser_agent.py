"""
================================================================================
PHASE 3 — SUMMARISER AGENT
================================================================================
Module  : agents/summariser_agent.py
Purpose : Takes the top-ranked papers from Phase 2 and generates plain-English
          summaries using Mistral 7B Instruct hosted on HuggingFace Inference API.

Why Mistral 7B Instruct?
    - Best-in-class open-source model for instruction following & summarisation
    - "Instruct" variant = fine-tuned to follow directions precisely
    - Runs on HuggingFace's free Inference API (no GPU needed on our end)
    - Fast enough for a daily newsletter pipeline (~5-10 sec per paper)

How summarisation works:
    1. We craft a carefully designed PROMPT for each paper
    2. The prompt instructs Mistral to explain the paper as if talking to
       a curious non-technical friend — no jargon, just clear insight
    3. Mistral returns a structured summary with: headline, what it is,
       why it matters, and a real-world analogy
    4. We parse and attach the summary back to the Paper object

HuggingFace Inference API:
    - Endpoint : https://api-inference.huggingface.co/models/{model_id}
    - Auth     : Bearer token (HF_TOKEN from .env)
    - Free tier: Rate limited but sufficient for daily newsletter (5 papers/day)
    - Docs     : https://huggingface.co/docs/api-inference

Author  : AI Research Digest Project
================================================================================
"""

import os
import re
import time
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

# Import our shared Paper model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.fetcher_arxiv import Paper

# Load environment variables from .env file (for local development)
# On HuggingFace Spaces, secrets are set via the Space settings UI
load_dotenv()


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class SummarisedPaper:
    """
    Extends a Paper with its generated plain-English summary.
    This is the final data object that gets passed to the newsletter agent.

    Attributes:
        paper        : The original Paper object (all metadata preserved)
        headline     : One punchy sentence that captures the paper's contribution
        what_it_does : 2-3 sentences explaining the research in plain English
        why_it_matters: 1-2 sentences on real-world impact / why you should care
        analogy      : A simple everyday analogy to make it click for non-tech readers
        summary_raw  : The full raw text returned by Mistral (kept for debugging)
    """
    paper         : Paper
    headline      : str
    what_it_does  : str
    why_it_matters: str
    analogy       : str
    summary_raw   : str = ""


# ── Summariser Agent ──────────────────────────────────────────────────────────

class SummariserAgent:
    """
    Calls HuggingFace Inference API to summarise each paper using
    Mistral 7B Instruct in plain, accessible English.

    Usage:
        agent = SummariserAgent()
        summarised_papers = agent.run(top_papers)

    Environment Variables Required:
        HF_TOKEN : Your HuggingFace API token.
                   Get one free at https://huggingface.co/settings/tokens
                   Set in .env file locally, or in HF Space secrets on deployment.
    """

    # Mistral 7B Instruct v0.3 — best open-source instruction-following model
    # that runs well on the HuggingFace free Inference API tier
    MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

    # HF Inference API base URL
    HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

    # How long to wait between API calls (seconds)
    # HF free tier is rate-limited — a small delay avoids 429 errors
    DELAY_BETWEEN_CALLS = 3

    # Max tokens for the generated summary
    # ~400 tokens ≈ 300 words — enough for a rich summary without being too long
    MAX_NEW_TOKENS = 700  # 700 tokens = enough for all 4 sections reliably

    # How many times to retry if the API returns an error
    MAX_RETRIES = 3

    def __init__(self):
        """
        Initialises the summariser and loads the HuggingFace API token.

        Raises:
            ValueError: If HF_TOKEN is not set in environment variables.
                        The API will not work without authentication.
        """
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is not set.\n"
                "  → For local dev: add HF_TOKEN=your_token to your .env file\n"
                "  → For HF Spaces: add it in Settings → Repository Secrets\n"
                "  → Get a free token at: https://huggingface.co/settings/tokens"
            )

        # Build the auth header — used in every API request
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        print(f"[Summariser Agent] Initialised with model: {self.MODEL_ID}")

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self, papers: list[Paper]) -> list[SummarisedPaper]:
        """
        Summarises a list of papers one by one and returns SummarisedPaper objects.

        We process papers sequentially (not in parallel) to:
        1. Respect HuggingFace free tier rate limits
        2. Avoid overwhelming the API with concurrent requests
        3. Give each paper the full model attention it deserves

        Args:
            papers: Top-ranked papers from FilterRankAgent (Phase 2)

        Returns:
            list[SummarisedPaper]: Papers with generated plain-English summaries.
                                   Papers that fail summarisation are included with
                                   a fallback summary so the newsletter still works.
        """
        print(f"\n[Summariser Agent] Summarising {len(papers)} papers with Mistral 7B...")
        print(f"[Summariser Agent] Estimated time: ~{len(papers) * 15} seconds\n")

        summarised = []

        for i, paper in enumerate(papers, 1):
            print(f"[Summariser Agent] ({i}/{len(papers)}) Summarising: {paper.title[:55]}...")

            # Generate the summary via API call
            summarised_paper = self._summarise_paper(paper)
            summarised.append(summarised_paper)

            # Polite delay between API calls to stay within rate limits
            # Skip delay after the last paper — no need to wait
            if i < len(papers):
                print(f"[Summariser Agent]   ⏳ Waiting {self.DELAY_BETWEEN_CALLS}s before next call...")
                time.sleep(self.DELAY_BETWEEN_CALLS)

        print(f"\n[Summariser Agent] ✅ Successfully summarised {len(summarised)} papers")
        return summarised

    # ── Core Summarisation ────────────────────────────────────────────────────

    def _summarise_paper(self, paper: Paper) -> SummarisedPaper:
        """
        Generates a plain-English summary for a single paper.

        Flow:
            1. Build a carefully crafted prompt
            2. Call Mistral 7B via HF Inference API (with retries)
            3. Parse the structured response into SummarisedPaper fields
            4. Return fallback summary if API fails

        Args:
            paper: A single Paper object with title + abstract

        Returns:
            SummarisedPaper: Paper with all summary fields populated
        """
        # Build the prompt
        prompt = self._build_prompt(paper)

        # Call the API with retry logic
        raw_response = self._call_api_with_retry(prompt)

        if raw_response:
            # Parse the structured response from Mistral
            parsed = self._parse_response(raw_response)
            print(f"[Summariser Agent]   ✅ Summary generated successfully")
        else:
            # API failed after all retries — use a fallback summary
            # This ensures the newsletter pipeline never breaks completely
            print(f"[Summariser Agent]   ⚠️  API failed — using fallback summary")
            parsed = self._build_fallback_summary(paper)

        return SummarisedPaper(
            paper          = paper,
            headline       = parsed["headline"],
            what_it_does   = parsed["what_it_does"],
            why_it_matters = parsed["why_it_matters"],
            analogy        = parsed["analogy"],
            summary_raw    = raw_response or "API call failed",
        )

    def _build_prompt(self, paper: Paper) -> str:
        """
        Crafts the prompt sent to Mistral 7B.

        Prompt design principles used here:
        1. ROLE       : Tell Mistral who it is ("science communicator")
                        — this primes the model's tone and vocabulary
        2. AUDIENCE   : Explicitly state "non-technical readers"
                        — Mistral will avoid jargon automatically
        3. STRUCTURE  : Request specific labelled sections
                        — makes parsing the response reliable
        4. CONSTRAINTS: "No bullet points", "plain English", "2-3 sentences"
                        — keeps the output consistent across all papers
        5. EXAMPLES   : The analogy instruction gives Mistral creative freedom
                        while keeping it grounded

        The [INST] ... [/INST] tags are Mistral's instruction format.
        They tell the model "this is a user instruction, respond to it."
        Using the wrong format degrades output quality significantly.

        Args:
            paper: Paper whose title and abstract will be included

        Returns:
            str: Fully formatted prompt string ready for the API
        """

        # Truncate abstract to 1200 chars to stay within Mistral's context window
        # Full abstracts can be very long — we only need the key information
        abstract_truncated = paper.abstract[:1200]

        prompt = (
            "<|system|>\n"
            "You are a science writer. Summarise AI research papers for non-technical readers.\n"
            "Always respond using EXACTLY these 4 labels on separate lines. Never skip any label.</s>\n"
            "<|user|>\n"
            f"Paper: {paper.title}\n\n"
            f"Abstract: {paper.abstract[:800]}\n\n"
            "Write exactly 4 sections using these labels:\n"
            "HEADLINE: one sentence summary\n"
            "WHAT IT DOES: 2 sentences explaining what was built or discovered\n"
            "WHY IT MATTERS: 1 sentence on real world impact\n"
            "ANALOGY: one sentence starting with \"Think of it like\"</s>\n"
            "<|assistant|>\n"
            "HEADLINE:"
        )

        return prompt

    def _call_api_with_retry(self, prompt: str) -> str | None:
        """
        Calls the HuggingFace Inference API with automatic retry logic.

        Why retry logic?
            - The HF free tier sometimes returns 503 (model loading) errors
            - Models on free tier are "cold" and need around 20s to warm up
            - Retrying after a short wait handles this gracefully

        Retry strategy:
            - Attempt 1: Immediate
            - Attempt 2: Wait 20 seconds (model warm-up time)
            - Attempt 3: Wait 30 seconds (longer backoff)

        Args:
            prompt: The fully formatted prompt string

        Returns:
            str  : The generated text if successful
            None : If all retries failed
        """
        # Retry wait times in seconds (one per retry attempt after the first)
        retry_waits = [20, 30]

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    self.HF_API_URL,
                    headers=self.headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            # Max tokens to generate (our summary target)
                            "max_new_tokens": self.MAX_NEW_TOKENS,

                            # Temperature: 0.7 = creative but not hallucinating
                            # Lower (0.3) = more factual but repetitive
                            # Higher (1.0) = very creative but unreliable
                            "temperature": 0.7,

                            # Top-p sampling: consider tokens comprising top 90%
                            # probability mass — good balance of quality/variety
                            "top_p": 0.9,

                            # Don't repeat the input prompt in the output
                            "return_full_text": False,

                            # Stop generating if these tokens appear
                            # Prevents the model from rambling after the summary
                            "stop": ["</s>", "[INST]"],
                        },
                        # Don't wait for model to load — we handle that in retry
                        "options": {"wait_for_model": True},
                    },
                    timeout=60,  # 60 second timeout per request
                )

                # ── Handle API Response Codes ─────────────────────────────────

                if response.status_code == 200:
                    # Success — extract generated text
                    result = response.json()

                    # HF API returns a list of generated texts
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                        if generated_text.strip():
                            return generated_text.strip()

                elif response.status_code == 503:
                    # Model is loading (cold start) — this is normal on free tier
                    print(f"[Summariser Agent]   Model loading (503)... attempt {attempt + 1}/{self.MAX_RETRIES}")

                elif response.status_code == 429:
                    # Rate limited — we're calling too fast
                    print(f"[Summariser Agent]   Rate limited (429)... attempt {attempt + 1}/{self.MAX_RETRIES}")

                else:
                    # Unexpected error — log it for debugging
                    print(f"[Summariser Agent]   API error {response.status_code}: {response.text[:100]}")

            except requests.Timeout:
                print(f"[Summariser Agent]   Request timed out — attempt {attempt + 1}/{self.MAX_RETRIES}")

            except requests.RequestException as e:
                print(f"[Summariser Agent]   Request error: {e} — attempt {attempt + 1}/{self.MAX_RETRIES}")

            # Wait before retrying (skip wait on last attempt)
            if attempt < self.MAX_RETRIES - 1:
                wait = retry_waits[min(attempt, len(retry_waits) - 1)]
                print(f"[Summariser Agent]   Retrying in {wait}s...")
                time.sleep(wait)

        # All retries exhausted
        return None

    # ── Response Parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw_text: str) -> dict:
        """
        V3 Parser — robust multi-strategy extraction from Mistral output.

        Strategy:
          1. Clean text  — strip markdown bold, numbered prefixes, preamble
          2. Label split — find known section labels in any case/format
          3. Para split  — fallback if no labels found (plain paragraph output)

        Handles all known Mistral output variations:
          - Clean uppercase     : HEADLINE: text
          - Lowercase           : headline: text
          - Mixed case          : Headline: / What It Does:
          - Bold markdown       : **HEADLINE:** text
          - Numbered            : 1. HEADLINE: text
          - With preamble       : "Sure! Here is the summary:\nHEADLINE:..."
          - No labels           : plain paragraphs (paragraph fallback)
          - Truncated output    : returns whatever sections were generated

        Args:
            raw_text: Raw string returned by Mistral 7B

        Returns:
            dict with keys: headline, what_it_does, why_it_matters, analogy
        """
        # Model is primed with "HEADLINE:" so prepend it back for parsing
        if not raw_text.upper().startswith("HEADLINE"):
            raw_text = "HEADLINE: " + raw_text
        print(f"[Summariser]   Raw ({len(raw_text)} chars): {raw_text[:120].strip()!r}")

        # ── Step 1: Clean the text ────────────────────────────────────────────
        text = raw_text.strip()
        text = re.sub(r'\*+', '', text)                                    # remove **bold**
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)        # remove "1. "
        text = re.sub(                                                       # remove preambles
            r'^(sure!?|here\'?s?|okay|of course|absolutely)[^\n]*\n',
            '', text, flags=re.IGNORECASE
        )
        text = text.strip()

        # ── Step 2: Label-based extraction ───────────────────────────────────
        # Maps output field → all label variants we accept
        label_map = {
            "headline":       ["HEADLINE"],
            "what_it_does":   ["WHAT IT DOES", "WHAT DOES IT DO"],
            "why_it_matters": ["WHY IT MATTERS", "WHY DOES IT MATTER"],
            "analogy":        ["ANALOGY"],
        }

        # Find position of every label variant in the text
        all_positions = []
        for field_key, variants in label_map.items():
            for variant in variants:
                pattern = rf'(?i)(?:^|\n)\s*{re.escape(variant)}\s*:'
                for m in re.finditer(pattern, text):
                    all_positions.append((m.start(), m.end(), field_key))

        all_positions.sort(key=lambda x: x[0])

        sections = {}
        if all_positions:
            for i, (start, end, field_key) in enumerate(all_positions):
                content_start = end
                content_end   = all_positions[i+1][0] if i+1 < len(all_positions) else len(text)
                chunk         = text[content_start:content_end].strip()
                chunk         = re.sub(r'\s+', ' ', chunk).strip()
                if chunk and field_key not in sections:
                    sections[field_key] = chunk

        # ── Step 3: Paragraph fallback ────────────────────────────────────────
        # If fewer than 2 sections found via labels, try paragraph splitting.
        # Some Mistral responses ignore labels and write plain paragraphs.
        if len(sections) < 2:
            paragraphs = [
                re.sub(r'\s+', ' ', p.strip())
                for p in re.split(r'\n\s*\n', text)
                if p.strip()
            ]
            if len(paragraphs) >= 4:
                sections["headline"]       = paragraphs[0]
                sections["what_it_does"]   = paragraphs[1]
                sections["why_it_matters"] = paragraphs[2]
                sections["analogy"]        = paragraphs[3]
            elif len(paragraphs) == 3:
                sections["headline"]       = paragraphs[0]
                sections["what_it_does"]   = paragraphs[1]
                sections["why_it_matters"] = paragraphs[2]
            elif paragraphs:
                sections["headline"]       = paragraphs[0]
                sections["what_it_does"]   = ' '.join(paragraphs[1:]) if len(paragraphs) > 1 else paragraphs[0]

        found = {k: bool(v) for k, v in sections.items()}
        print(f"[Summariser]   Sections found: {found}")

        return {
            "headline"      : sections.get("headline",       "") or "New AI Research Breakthrough",
            "what_it_does"  : sections.get("what_it_does",   "") or raw_text[:300].strip(),
            "why_it_matters": sections.get("why_it_matters", "") or "This research advances the state of AI.",
            "analogy"       : sections.get("analogy",        "") or "Think of it like teaching a computer a smarter way to solve problems.",
        }

    def _build_fallback_summary(self, paper: Paper) -> dict:
        """
        Builds a basic summary from the paper's own abstract when the API fails.

        This is a safety net — the newsletter can still go out even if Mistral
        is unavailable. The fallback uses the first 300 chars of the abstract
        as a "what it does" and generates generic but accurate other fields.

        Args:
            paper: The Paper object whose abstract we'll use as fallback

        Returns:
            dict with keys: headline, what_it_does, why_it_matters, analogy
        """
        return {
            "headline"      : paper.title,
            "what_it_does"  : paper.abstract[:350].strip() + "..." if len(paper.abstract) > 350 else paper.abstract,
            "why_it_matters": "This research contributes to the advancement of artificial intelligence.",
            "analogy"       : "Think of it like researchers finding a better way to solve a complex puzzle that computers face.",
        }


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone test for the Summariser Agent.

    To run this test you need a valid HF_TOKEN in your .env file.
    The test uses a single mock paper to verify the full API call,
    prompt construction, and response parsing pipeline.

    If you don't have an HF_TOKEN yet:
        1. Go to https://huggingface.co/settings/tokens
        2. Click "New token" → Role: "Read" → Copy the token
        3. Create a .env file in the project root:
               HF_TOKEN=hf_your_token_here
    """
    from agents.fetcher_arxiv import Paper

    # Single test paper — rich abstract for a good summary test
    test_paper = Paper(
        paper_id      = "test_001",
        title         = "Large Language Models for Reasoning: A Comprehensive Survey",
        authors       = ["Alice Smith", "Bob Jones"],
        abstract      = (
            "Large language models (LLMs) have demonstrated remarkable reasoning "
            "capabilities across a variety of tasks including mathematical problem "
            "solving, logical inference, and commonsense reasoning. In this survey, "
            "we comprehensively review recent advances in chain-of-thought prompting, "
            "reinforcement learning from human feedback (RLHF), and emergent abilities "
            "in transformer-based foundation models. We find that models trained with "
            "instruction tuning and RLHF show significantly improved reasoning compared "
            "to base models, with GPT-4 class models achieving near-human performance "
            "on standardised benchmarks. We discuss alignment challenges, safety "
            "considerations, and future research directions for building more capable "
            "and trustworthy reasoning systems."
        ),
        published_date= "2025-03-09",
        url           = "https://arxiv.org/abs/test_001",
        source        = "arxiv",
        categories    = ["cs.AI"],
    )

    print("="*60)
    print("  SUMMARISER AGENT — TEST RUN")
    print("="*60)

    try:
        agent    = SummariserAgent()
        results  = agent.run([test_paper])
        result   = results[0]

        print(f"\n📰 HEADLINE     : {result.headline}")
        print(f"\n📖 WHAT IT DOES : {result.what_it_does}")
        print(f"\n💡 WHY IT MATTERS: {result.why_it_matters}")
        print(f"\n🔍 ANALOGY      : {result.analogy}")

    except ValueError as e:
        # HF_TOKEN not set — show helpful setup instructions
        print(f"\n⚠️  Setup required: {e}")
