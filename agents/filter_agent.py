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
    Step 3b— TOPICS   : Keep only papers matching the user's chosen topics
                        (with graceful fallback if too few match)
    Step 4 — SCORE    : Score each paper using a multi-factor ranking formula
    Step 5 — SORT     : Sort by score descending
    Step 6 — TRIM     : Return only the top N papers

Scoring Formula (see _score_paper() for full details):
    Score = recency_score + title_quality_score + abstract_quality_score
              + keyword_relevance_score + topic_bonus

Author  : AI Research Digest Project
================================================================================
"""

import re
from datetime import datetime, timezone
from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.fetcher_arxiv import Paper


# ── Topic Definitions ─────────────────────────────────────────────────────────
#
# Each topic maps to a list of keywords. A paper matches a topic if ANY of its
# keywords appear in the paper's title or abstract (case-insensitive).
#
# Design notes:
#   - Keywords intentionally overlap across topics (e.g. "agent" appears in
#     both Agents & Robotics and LLMs). That's fine — a paper matching multiple
#     topics is MORE relevant, not less.
#   - Keep keywords lowercase — matching is done on lowercased text.
#   - Phrase keywords ("chain of thought") work the same as single words.

TOPIC_KEYWORDS = {
    "🧠 LLMs & NLP": [
        "large language model", "llm", "language model", "gpt", "bert",
        "transformer", "instruction tuning", "fine-tuning", "finetuning",
        "rlhf", "reinforcement learning from human feedback",
        "chain of thought", "in-context learning", "prompt", "tokenizer",
        "text generation", "summarisation", "summarization", "translation",
        "question answering", "dialogue", "chatbot", "speech recognition",
        "named entity", "sentiment", "embedding", "retrieval augmented",
        "rag", "hallucination", "alignment", "reasoning",
    ],
    "👁️ Computer Vision": [
        "computer vision", "image recognition", "object detection",
        "image segmentation", "diffusion model", "diffusion", "stable diffusion",
        "text to image", "text-to-image", "image generation", "video generation",
        "visual", "vision language", "multimodal", "vit", "vision transformer",
        "convolutional", "cnn", "scene understanding", "depth estimation",
        "optical flow", "3d reconstruction", "nerf", "image editing",
        "face recognition", "pose estimation",
    ],
    "🤖 Agents & Robotics": [
        "agent", "autonomous agent", "multi-agent", "tool use", "tool-use",
        "robotics", "robot learning", "robot", "planning", "task planning",
        "decision making", "autonomous", "embodied", "navigation",
        "manipulation", "reinforcement learning", "reward", "policy",
        "environment", "simulation", "world model", "agentic",
        "code generation", "code execution",
    ],
    "🛡️ Safety & Alignment": [
        "safety", "alignment", "hallucination", "bias", "fairness",
        "toxicity", "red teaming", "red-teaming", "adversarial",
        "interpretability", "explainability", "transparency",
        "robustness", "privacy", "watermark", "copyright",
        "misinformation", "disinformation", "jailbreak",
        "constitutional ai", "value alignment", "trustworthy",
        "responsible ai", "ethical",
    ],
    "🔬 Science & Healthcare": [
        "medical", "healthcare", "clinical", "drug discovery", "protein",
        "genomics", "biology", "chemistry", "molecule", "disease",
        "diagnosis", "radiology", "pathology", "ehr", "electronic health",
        "scientific discovery", "materials science", "physics simulation",
        "climate", "weather", "biology", "neuroscience",
    ],
    "⚙️ ML Foundations": [
        "architecture", "attention mechanism", "neural architecture search",
        "training", "optimisation", "optimization", "gradient",
        "batch normalisation", "batch normalization", "dropout",
        "overfitting", "generalisation", "generalization",
        "few-shot", "zero-shot", "meta-learning", "transfer learning",
        "self-supervised", "contrastive learning", "knowledge distillation",
        "quantization", "pruning", "efficient", "scalable", "benchmark",
        "evaluation", "dataset",
    ],
}

# All valid topic names — used for validation
ALL_TOPICS = list(TOPIC_KEYWORDS.keys())

# ── General High-Impact Keywords (used for scoring, not topic filtering) ──────
HIGH_IMPACT_KEYWORDS = [
    "large language model", "llm", "transformer", "diffusion", "multimodal",
    "foundation model", "vision language", "generative", "gpt", "bert",
    "reasoning", "alignment", "fine-tuning", "rlhf", "reinforcement learning",
    "chain of thought", "agent", "retrieval augmented", "rag", "benchmark",
    "hallucination", "emergent", "in-context learning", "prompt",
    "code generation", "text to image", "speech recognition", "robotics",
    "autonomous", "instruction following", "safety", "bias", "evaluation",
]

MIN_ABSTRACT_LENGTH = 100
MIN_TITLE_LENGTH    = 10

# ── Topic Filtering Fallback Thresholds ───────────────────────────────────────
#
# Edge case: user selects niche topics (e.g. only "🔬 Science & Healthcare")
# and there are very few or zero matching papers on a given day.
#
# Strategy: progressive relaxation
#   Level 0 — strict:   only papers matching at least 1 selected topic keyword
#   Level 1 — relaxed:  if < MIN_PAPERS_STRICT papers, fall back to full pool
#
# MIN_PAPERS_STRICT: minimum papers required to use strict topic filtering.
# If fewer than this match, we fall back to the full unfiltered pool so the
# user still gets a useful digest rather than an empty or near-empty email.
MIN_PAPERS_STRICT = 3   # Must have at least 3 topic-matching papers to filter


# ── Filter & Rank Agent ───────────────────────────────────────────────────────

class FilterRankAgent:
    """
    Merges, deduplicates, filters, and ranks research papers from multiple
    sources to produce a high-quality shortlist for the newsletter.

    Usage:
        agent = FilterRankAgent(top_n=5)
        top_papers = agent.run(arxiv_papers, hf_papers)

        # With topic filtering:
        agent = FilterRankAgent(top_n=5, topics=["🧠 LLMs & NLP", "🛡️ Safety & Alignment"])
        top_papers = agent.run(arxiv_papers, hf_papers)

    Args:
        top_n  (int) : How many papers to return after ranking. Default 5.
        topics (list): Topic names from TOPIC_KEYWORDS to filter by.
                       If None or empty, all topics are included (no filtering).
    """

    def __init__(self, top_n: int = 5, topics: list = None):
        self.top_n  = top_n
        # Normalise topics — None or empty list both mean "show everything"
        self.topics = topics if topics else ALL_TOPICS

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self, *paper_lists: list[Paper]) -> list[Paper]:
        """
        Main pipeline: merge → dedupe → quality filter → topic filter
        → score → sort → balanced select.

        Returns:
            list[Paper]: Top N ranked papers matching the user's topic prefs.
        """
        print(f"\n[Filter Agent] Starting filter & rank pipeline...")
        print(f"[Filter Agent] Topics: {', '.join(self.topics)}")

        # Step 1: Merge
        merged = []
        for paper_list in paper_lists:
            merged.extend(paper_list)
        print(f"[Filter Agent] Step 1 — Merged: {len(merged)} total papers")

        # Step 2: Deduplicate
        deduped = self._deduplicate(merged)
        print(f"[Filter Agent] Step 2 — Deduplicated: {len(deduped)} unique papers")

        # Step 3: Quality filter
        filtered = self._filter(deduped)
        print(f"[Filter Agent] Step 3 — After quality filter: {len(filtered)} papers")

        # Step 3b: Topic filter (with fallback)
        topic_filtered, used_fallback = self._filter_by_topics(filtered)
        if used_fallback:
            print(f"[Filter Agent] Step 3b — ⚠️  Topic filter: only {len(topic_filtered)} matched, using full pool")
        else:
            print(f"[Filter Agent] Step 3b — Topic filter: {len(topic_filtered)} papers match selected topics")

        # Step 4 & 5: Score and sort
        scored = self._score_and_sort(topic_filtered)
        print(f"[Filter Agent] Step 4 — Scored and sorted {len(scored)} papers")

        # Step 6: Balanced trim
        top_papers = self._balanced_select(scored, self.top_n)
        print(f"[Filter Agent] ✅ Final shortlist: Top {len(top_papers)} papers selected\n")

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

    # ── Step 3b: Topic Filter ─────────────────────────────────────────────────

    def _filter_by_topics(self, papers: list[Paper]) -> tuple[list[Paper], bool]:
        """
        Filters papers to only those matching the user's selected topics.

        A paper MATCHES if ANY keyword from ANY selected topic appears in
        either its title or abstract (case-insensitive).

        Edge cases handled:
        ┌─────────────────────────────────────────────────────────────────────┐
        │ Edge Case 1 — Too few matches (< MIN_PAPERS_STRICT)                 │
        │   Cause: User picked a niche topic (e.g. only "🔬 Science")         │
        │           and today had very few papers on that topic.               │
        │   Fix:  Fall back to the full unfiltered pool so the user still     │
        │          receives a useful digest. A warning is shown in the UI.    │
        ├─────────────────────────────────────────────────────────────────────┤
        │ Edge Case 2 — All topics selected (default)                         │
        │   Cause: User hasn't changed the default (all topics ticked).       │
        │   Fix:  Skip filtering entirely — return full pool unchanged.        │
        │          Avoids unnecessary keyword scanning when nothing to filter. │
        ├─────────────────────────────────────────────────────────────────────┤
        │ Edge Case 3 — Empty topics list                                      │
        │   Cause: Bug in caller, or user deselected everything somehow.      │
        │   Fix:  Treat same as "all selected" — return full pool.            │
        │          Prevents an empty digest from being sent.                  │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            papers: Quality-filtered list of papers

        Returns:
            tuple:
                list[Paper] — Papers that matched (or full pool if fallback)
                bool        — True if fallback was used, False if normal filter
        """
        # Edge Case 2 & 3: if all topics selected or none selected, skip filter
        if not self.topics or set(self.topics) == set(ALL_TOPICS):
            return papers, False

        # Build a flat set of all keywords for selected topics
        selected_keywords = set()
        for topic in self.topics:
            if topic in TOPIC_KEYWORDS:
                selected_keywords.update(TOPIC_KEYWORDS[topic])

        if not selected_keywords:
            return papers, False

        # Filter: keep papers where title or abstract contains any keyword
        matched = []
        for paper in papers:
            text = (paper.title + " " + paper.abstract).lower()
            if any(kw in text for kw in selected_keywords):
                matched.append(paper)

        # Edge Case 1: too few matches — fall back to full pool
        if len(matched) < MIN_PAPERS_STRICT:
            print(
                f"[Filter Agent]   ⚠️  Only {len(matched)} papers matched topics "
                f"{self.topics} — falling back to full pool of {len(papers)} papers"
            )
            return papers, True

        return matched, False



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

        # ── Factor 5: Topic relevance bonus (max 15 points) ──────────────────
        # Papers that match the user's selected topics get a bonus to push them
        # higher in the ranking relative to off-topic papers that slipped through.
        # This matters most in fallback mode — when the full pool is used because
        # too few topic-specific papers were found, we still want on-topic papers
        # to rank above off-topic ones.
        if set(self.topics) != set(ALL_TOPICS):  # Only apply if user filtered topics
            selected_keywords = set()
            for topic in self.topics:
                if topic in TOPIC_KEYWORDS:
                    selected_keywords.update(TOPIC_KEYWORDS[topic])
            text = (paper.title + " " + paper.abstract).lower()
            topic_hits = sum(1 for kw in selected_keywords if kw in text)
            score += min(topic_hits * 3, 15)   # Cap at 15 pts

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
