"""
================================================================================
PHASE 2 — FILTER & RANK AGENT
================================================================================
Module  : agents/filter_agent.py
Purpose : Takes the raw list of papers fetched by both fetcher agents (arXiv +
          HuggingFace) and produces a clean, ranked shortlist of the TOP N most
          relevant and impactful papers to be summarised and sent in the newsletter.

Why this step matters:
    - arXiv alone publishes 200-300 AI papers per day
    - Without filtering, the newsletter would be overwhelming and noisy
    - We want only the BEST papers — ones that are genuinely worth reading

What this module does (in order):
    Step 1 — MERGE    : Combine papers from both sources into one list
    Step 2 — DEDUPE   : Remove duplicate papers (same paper on arXiv + HF)
    Step 3 — FILTER   : Remove papers with missing/poor quality data
    Step 4 — SCORE    : Score each paper using a multi-factor ranking formula
    Step 5 — SORT     : Sort by score descending
    Step 6 — TRIM     : Return only the top N papers

Scoring Formula (see _score_paper() for full details):
    Score = recency_score + title_quality_score + abstract_quality_score
              + keyword_relevance_score + source_bonus

Author  : AI Research Digest Project
================================================================================
"""

import re
from datetime import datetime, timezone
from dataclasses import dataclass, field

# Import the shared Paper dataclass from our arXiv fetcher
# (both fetchers use the same Paper model — that's by design)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.fetcher_arxiv import Paper


# ── Constants ─────────────────────────────────────────────────────────────────

# High-impact AI keywords — papers mentioning these are likely more relevant
# to the current state of AI and more interesting to our readers.
# Feel free to expand this list as AI evolves.
HIGH_IMPACT_KEYWORDS = [
    # Model types & architectures
    "large language model", "llm", "transformer", "diffusion", "multimodal",
    "foundation model", "vision language", "generative", "gpt", "bert",

    # Hot research areas
    "reasoning", "alignment", "fine-tuning", "rlhf", "reinforcement learning",
    "chain of thought", "agent", "retrieval augmented", "rag", "benchmark",
    "hallucination", "emergent", "in-context learning", "prompt",

    # Applications getting lots of attention
    "code generation", "text to image", "speech recognition", "robotics",
    "autonomous", "instruction following", "safety", "bias", "evaluation",
]

# Minimum quality thresholds — papers below these are filtered out
MIN_ABSTRACT_LENGTH = 100    # Characters. Very short = incomplete/bad data
MIN_TITLE_LENGTH    = 10     # Characters. Very short titles are usually bad data


# ── Filter & Rank Agent ───────────────────────────────────────────────────────

class FilterRankAgent:
    """
    Merges, deduplicates, filters, and ranks research papers from multiple
    sources to produce a high-quality shortlist for the newsletter.

    Usage:
        agent = FilterRankAgent(top_n=5)
        top_papers = agent.run(arxiv_papers, hf_papers)

    Args:
        top_n (int): How many papers to return after ranking. Default is 5.
                     5 is a good number for a daily newsletter — enough to be
                     informative, not so many it overwhelms the reader.
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self, *paper_lists: list[Paper]) -> list[Paper]:
        """
        Main pipeline method. Accepts any number of paper lists (one per source)
        and returns the final ranked shortlist.

        Args:
            *paper_lists: Variable number of paper lists.
                          e.g. run(arxiv_papers, hf_papers)
                          or   run(arxiv_papers, hf_papers, another_source)

        Returns:
            list[Paper]: Top N ranked papers, best first.

        Example:
            agent = FilterRankAgent(top_n=5)
            results = agent.run(arxiv_papers, hf_papers)
        """
        print(f"\n[Filter Agent] Starting filter & rank pipeline...")

        # ── Step 1: Merge all paper lists into one ────────────────────────────
        # Flatten the list-of-lists into a single list
        merged = []
        for paper_list in paper_lists:
            merged.extend(paper_list)

        print(f"[Filter Agent] Step 1 — Merged: {len(merged)} total papers")

        # ── Step 2: Deduplicate ───────────────────────────────────────────────
        # The same paper often appears on both arXiv and HuggingFace Papers.
        # We keep the HuggingFace version when there's a conflict because
        # HF Papers are community-curated (higher signal).
        deduped = self._deduplicate(merged)
        print(f"[Filter Agent] Step 2 — Deduplicated: {len(deduped)} unique papers")

        # ── Step 3: Filter out low-quality entries ────────────────────────────
        filtered = self._filter(deduped)
        print(f"[Filter Agent] Step 3 — After quality filter: {len(filtered)} papers")

        # ── Step 4 & 5: Score and sort ────────────────────────────────────────
        scored = self._score_and_sort(filtered)
        print(f"[Filter Agent] Step 4 — Scored and sorted {len(scored)} papers")

        # ── Step 6: Balanced trim — guarantee source diversity ───────────────
        top_papers = self._balanced_select(scored, self.top_n)
        print(f"[Filter Agent] ✅ Final shortlist: Top {len(top_papers)} papers selected\n")

        # Print a summary of what was selected
        self._print_summary(top_papers)

        return top_papers

    # ── Step 2: Deduplication ─────────────────────────────────────────────────

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """
        Removes duplicate papers that appear across multiple sources.

        Deduplication strategy:
            1. Exact ID match  — same arXiv ID appears in both sources
            2. Title match     — very similar titles (handles slight variations)

        When a duplicate is found, we KEEP the HuggingFace version because
        HF Papers are hand-curated by the community, meaning the paper was
        notable enough for someone to submit it there.

        Args:
            papers: Raw merged list with potential duplicates

        Returns:
            list[Paper]: Deduplicated list
        """
        seen_ids    = {}   # Maps normalised paper_id → Paper
        seen_titles = {}   # Maps normalised title → Paper

        result = []

        for paper in papers:
            # Normalise the paper ID for comparison
            # arXiv IDs look like "2401.12345" or "hf_2401.12345"
            # We strip the "hf_" prefix to compare apples to apples
            normalised_id = paper.paper_id.replace("hf_", "").strip().lower()

            # Normalise title: lowercase, remove punctuation, collapse spaces
            normalised_title = re.sub(r"[^\w\s]", "", paper.title.lower())
            normalised_title = re.sub(r"\s+", " ", normalised_title).strip()

            # Check if we've seen this paper before (by ID or title)
            if normalised_id in seen_ids:
                # Duplicate found by ID — prefer arXiv (richer metadata)
                if paper.source == "arxiv":
                    idx = result.index(seen_ids[normalised_id])
                    result[idx] = paper
                    seen_ids[normalised_id] = paper
                # Otherwise keep what we already have
                continue

            if normalised_title in seen_titles:
                # Duplicate found by title — prefer arXiv
                if paper.source == "arxiv":
                    idx = result.index(seen_titles[normalised_title])
                    result[idx] = paper
                    seen_titles[normalised_title] = paper
                continue

            # Not a duplicate — add to result
            seen_ids[normalised_id]       = paper
            seen_titles[normalised_title] = paper
            result.append(paper)

        return result

    # ── Step 3: Quality Filter ────────────────────────────────────────────────

    def _filter(self, papers: list[Paper]) -> list[Paper]:
        """
        Removes papers that don't meet minimum quality standards.

        A paper is REMOVED if:
            - Title is missing or too short (likely a scraping error)
            - Abstract is missing or too short (not enough info to summarise)
            - Title contains junk patterns (e.g. just numbers, all caps noise)

        Args:
            papers: Deduplicated list of papers

        Returns:
            list[Paper]: Only papers that pass all quality checks
        """
        filtered = []

        for paper in papers:
            # Check 1: Title must exist and meet minimum length
            if not paper.title or len(paper.title.strip()) < MIN_TITLE_LENGTH:
                print(f"  [Filter] ✗ Removed (short title): '{paper.title[:40]}'")
                continue

            # Check 2: Abstract must exist and meet minimum length
            # Without a decent abstract, Mistral 7B can't generate a good summary
            if not paper.abstract or len(paper.abstract.strip()) < MIN_ABSTRACT_LENGTH:
                print(f"  [Filter] ✗ Removed (missing/short abstract): '{paper.title[:40]}'")
                continue

            # Check 3: Title shouldn't be all uppercase (scraping noise)
            if paper.title == paper.title.upper() and len(paper.title) > 20:
                print(f"  [Filter] ✗ Removed (all-caps title, likely noise): '{paper.title[:40]}'")
                continue

            # Passed all checks
            filtered.append(paper)

        return filtered

    # ── Step 4 & 5: Scoring & Sorting ────────────────────────────────────────

    def _score_and_sort(self, papers: list[Paper]) -> list[Paper]:
        """
        Scores each paper using a multi-factor formula, then sorts
        highest score first.

        See _score_paper() for the full scoring breakdown.

        Args:
            papers: Filtered list of papers

        Returns:
            list[Paper]: Same papers, sorted by score descending
        """
        # Score each paper and attach the score temporarily for sorting
        scored_pairs = []
        for paper in papers:
            score = self._score_paper(paper)
            scored_pairs.append((score, paper))

        # Sort by score descending (highest score = best paper = goes first)
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        # Return just the papers (discard scores — they're internal only)
        return [paper for score, paper in scored_pairs]

    def _score_paper(self, paper: Paper) -> float:
        """
        Calculates a relevance/impact score for a single paper.

        Scoring breakdown (max possible score ≈ 100):
        ┌─────────────────────────────┬──────────┬─────────────────────────────┐
        │ Factor                      │ Max Pts  │ Why it matters              │
        ├─────────────────────────────┼──────────┼─────────────────────────────┤
        │ Recency                     │   30     │ Newer = more relevant       │
        │ Keyword relevance (title)   │   25     │ Title signals topic quality │
        │ Keyword relevance (abstract)│   20     │ Abstract confirms substance │
        │ Abstract length             │   15     │ Longer = more info to use   │
        │ Source bonus (HF)           │   10     │ HF = community-curated      │
        └─────────────────────────────┴──────────┴─────────────────────────────┘

        Args:
            paper: A single Paper object

        Returns:
            float: Score value (higher = better)
        """
        score = 0.0

        # ── Factor 1: Recency (max 30 points) ────────────────────────────────
        # More recent papers score higher. We use a decay formula:
        # Papers from today = 30 pts, yesterday = 20 pts, older = scaled down
        try:
            # Parse the published date — handle both full datetime and date-only
            date_str = paper.published_date.split(" ")[0]  # Take "YYYY-MM-DD" part
            pub_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now      = datetime.now(timezone.utc)
            age_days = (now - pub_date).days

            if age_days == 0:
                score += 30      # Published today
            elif age_days == 1:
                score += 20      # Published yesterday
            elif age_days <= 3:
                score += 10      # Published within 3 days
            elif age_days <= 7:
                score += 5       # Published within a week
            # Older than a week = 0 recency points
        except (ValueError, AttributeError):
            # If date parsing fails, give a neutral score (don't penalise)
            score += 10

        # ── Factor 2: Keyword relevance in TITLE (max 25 points) ─────────────
        # The title is the most important signal — if hot keywords appear in
        # the title, this is almost certainly a relevant paper
        title_lower = paper.title.lower()
        title_keyword_hits = sum(
            1 for kw in HIGH_IMPACT_KEYWORDS if kw in title_lower
        )
        # Cap at 5 hits × 5 points each = max 25 points
        score += min(title_keyword_hits * 5, 25)

        # ── Factor 3: Keyword relevance in ABSTRACT (max 20 points) ──────────
        # Abstract confirms the paper actually covers the topic (not just title)
        abstract_lower = paper.abstract.lower()
        abstract_keyword_hits = sum(
            1 for kw in HIGH_IMPACT_KEYWORDS if kw in abstract_lower
        )
        # Cap at 4 hits × 5 points each = max 20 points
        score += min(abstract_keyword_hits * 5, 20)

        # ── Factor 4: Abstract length (max 15 points) ─────────────────────────
        # Longer abstracts give the summariser more material to work with.
        # We reward papers with detailed abstracts, up to a sensible limit.
        abstract_len = len(paper.abstract)
        if abstract_len >= 1000:
            score += 15     # Very detailed abstract
        elif abstract_len >= 600:
            score += 10     # Good length
        elif abstract_len >= 300:
            score += 5      # Acceptable
        # Below 300 chars (but above MIN_ABSTRACT_LENGTH) = 0 length bonus

        # ── Factor 5: Source bonus removed — scoring is now neutral ─────────
        # Diversity is enforced by _balanced_select in the trim step.

        return score

    # ── Utility ───────────────────────────────────────────────────────────────

    def _balanced_select(self, scored_papers, top_n):
        """
        Selects top N papers ensuring at least 1 from each source.
        Prevents all 5 papers coming from the same provider.
        Fills remaining slots with best-scoring papers overall.
        """
        arxiv_papers = [p for p in scored_papers if p.source == "arxiv"]
        hf_papers    = [p for p in scored_papers if p.source == "huggingface"]

        selected = []
        used_ids = set()

        def add_paper(paper):
            if paper.paper_id not in used_ids:
                selected.append(paper)
                used_ids.add(paper.paper_id)

        # Guarantee at least 1 from each source
        if arxiv_papers:
            add_paper(arxiv_papers[0])
        if hf_papers:
            add_paper(hf_papers[0])

        # Fill remaining with best across both sources
        for paper in scored_papers:
            if len(selected) >= top_n:
                break
            add_paper(paper)

        arxiv_count = sum(1 for p in selected if p.source == "arxiv")
        hf_count    = sum(1 for p in selected if p.source == "huggingface")
        print(f"[Filter Agent]   Source mix — arXiv: {arxiv_count} | HuggingFace: {hf_count}")
        return selected

    def _print_summary(self, papers: list[Paper]) -> None:
        """
        Prints a readable summary of the selected top papers to the console.
        Useful for debugging and monitoring the pipeline.

        Args:
            papers: The final shortlisted papers
        """
        print(f"{'='*65}")
        print(f"  📰 TOP {len(papers)} PAPERS SELECTED FOR NEWSLETTER")
        print(f"{'='*65}")
        for i, p in enumerate(papers, 1):
            # Truncate long titles for display
            title_display = p.title[:55] + "..." if len(p.title) > 55 else p.title
            print(f"\n  [{i}] {title_display}")
            print(f"       Source : {p.source.upper()}")
            print(f"       Date   : {p.published_date}")
            print(f"       URL    : {p.url}")
        print(f"\n{'='*65}\n")


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone test using mock Paper objects so we can verify the filtering
    and scoring logic without needing a live internet connection.

    To test with real data, use:
        from agents.fetcher_arxiv import ArxivFetcherAgent
        from agents.fetcher_hf import HuggingFaceFetcherAgent
        arxiv_papers = ArxivFetcherAgent().fetch()
        hf_papers    = HuggingFaceFetcherAgent().fetch()
        top_papers   = FilterRankAgent(top_n=5).run(arxiv_papers, hf_papers)
    """

    # Create mock papers to test filtering and scoring logic
    mock_papers = [
        Paper(
            paper_id="2401.00001",
            title="Large Language Models for Reasoning: A Comprehensive Survey",
            authors=["Alice Smith", "Bob Jones"],
            abstract=(
                "Large language models (LLMs) have demonstrated remarkable reasoning "
                "capabilities across a variety of tasks. In this survey, we review "
                "recent advances in chain-of-thought prompting, reinforcement learning "
                "from human feedback (RLHF), and emergent abilities in transformer-based "
                "foundation models. We evaluate benchmark performance and discuss "
                "alignment challenges for generative AI systems deployed at scale. "
                "Our analysis covers over 150 papers published in 2023-2024."
            ),
            published_date="2025-03-09",  # Today — should score high on recency
            url="https://arxiv.org/abs/2401.00001",
            source="arxiv",
            categories=["cs.AI"],
        ),
        Paper(
            paper_id="hf_2401.00001",    # DUPLICATE of above — HF version
            title="Large Language Models for Reasoning: A Comprehensive Survey",
            authors=["Alice Smith", "Bob Jones"],
            abstract=(
                "Large language models (LLMs) have demonstrated remarkable reasoning "
                "capabilities. This HuggingFace version should be kept over arXiv "
                "version during deduplication because HF papers are community curated."
            ),
            published_date="2025-03-09",
            url="https://huggingface.co/papers/2401.00001",
            source="huggingface",         # HF version — should win dedup
            categories=["AI/ML"],
        ),
        Paper(
            paper_id="2401.00002",
            title="Diffusion Models for Text-to-Image Generation",
            authors=["Carol White"],
            abstract=(
                "We present a novel diffusion-based generative model for high-fidelity "
                "text to image synthesis. Our multimodal architecture uses a vision "
                "language transformer backbone trained on 5 billion image-text pairs. "
                "The model achieves state-of-the-art results on standard benchmarks "
                "including FID and CLIP scores, demonstrating strong instruction "
                "following for diverse prompt styles."
            ),
            published_date="2025-03-08",  # Yesterday
            url="https://arxiv.org/abs/2401.00002",
            source="arxiv",
            categories=["cs.CV"],
        ),
        Paper(
            paper_id="2401.00003",
            title="X",                   # TOO SHORT — should be filtered out
            authors=["Unknown"],
            abstract="Short.",           # TOO SHORT — should be filtered out
            published_date="2025-03-09",
            url="https://arxiv.org/abs/2401.00003",
            source="arxiv",
            categories=["cs.AI"],
        ),
        Paper(
            paper_id="2401.00004",
            title="Autonomous AI Agents with Tool Use and Code Generation",
            authors=["Dan Lee", "Eva Brown", "Frank Zhang"],
            abstract=(
                "We introduce a framework for autonomous AI agents that can use external "
                "tools, write and execute code, and perform multi-step reasoning to solve "
                "complex tasks. The agent uses a fine-tuned LLaMA foundation model with "
                "reinforcement learning to learn effective tool-use policies. Evaluation "
                "on agent benchmarks shows significant improvement over baseline LLMs."
            ),
            published_date="2025-03-07",  # 2 days ago
            url="https://arxiv.org/abs/2401.00004",
            source="arxiv",
            categories=["cs.AI"],
        ),
        Paper(
            paper_id="2401.00005",
            title="A Study on Numerical Optimisation",  # Low keyword relevance
            authors=["Grace Kim"],
            abstract=(
                "This paper examines classical numerical optimisation methods including "
                "gradient descent, Newton methods, and quasi-Newton approaches. We provide "
                "theoretical convergence analysis and empirical comparisons on standard "
                "optimisation benchmarks. Results show that second-order methods outperform "
                "first-order methods on smooth convex objectives in terms of convergence rate."
            ),
            published_date="2025-03-09",
            url="https://arxiv.org/abs/2401.00005",
            source="arxiv",
            categories=["math.OC"],
        ),
    ]

    print("Running FilterRankAgent test with mock papers...")
    print(f"Input: {len(mock_papers)} papers (includes 1 duplicate + 1 low quality)\n")

    agent      = FilterRankAgent(top_n=3)
    top_papers = agent.run(mock_papers)

    print(f"Output: {len(top_papers)} top papers returned")
    print("\nExpected order:")
    print("  [1] LLM Reasoning Survey (HF version — won dedup + high keywords + today)")
    print("  [2] Autonomous AI Agents  (high keywords)")
    print("  [3] Diffusion Models      (good keywords + yesterday)")
    print("  'X' paper should be filtered. Numerical Optimisation should rank last.")
