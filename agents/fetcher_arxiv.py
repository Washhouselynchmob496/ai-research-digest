"""
AGENT: ArXiv Fetcher
PURPOSE: Fetches the latest AI research papers from arXiv API
         published in the last 24 hours (or N hours back)
API DOCS: https://arxiv.org/help/api
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """Represents a single research paper."""
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: str
    url: str
    source: str          # "arxiv" or "huggingface"
    categories: list[str]


# ── ArXiv Fetcher Agent ───────────────────────────────────────────────────────

class ArxivFetcherAgent:
    """
    Fetches recent AI papers from the arXiv public API.
    
    ArXiv has a free, no-auth-required API — perfect for open-source projects.
    We query the 'cs.AI', 'cs.LG', 'cs.CL' categories (AI, ML, NLP).
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    # AI-related arXiv category codes
    AI_CATEGORIES = [
        "cs.AI",   # Artificial Intelligence
        "cs.LG",   # Machine Learning
        "cs.CL",   # Computation and Language (NLP)
        "cs.CV",   # Computer Vision
        "stat.ML", # Stats / ML
    ]

    def __init__(self, max_results: int = 20, hours_back: int = 24):
        """
        Args:
            max_results: How many papers to fetch per query (we filter after)
            hours_back:  How far back to look (default = last 24 hours)
        """
        self.max_results = max_results
        self.hours_back = hours_back

    def fetch(self) -> list[Paper]:
        """
        Main entry point — fetches and returns a list of Paper objects.
        Queries each AI category and deduplicates by paper ID.
        """
        print(f"[ArXiv Agent] Fetching papers from last {self.hours_back} hours...")

        all_papers = {}  # dict keyed by paper_id to auto-deduplicate

        for category in self.AI_CATEGORIES:
            papers = self._fetch_category(category)
            for paper in papers:
                all_papers[paper.paper_id] = paper  # deduplicate

        result = list(all_papers.values())
        print(f"[ArXiv Agent] ✅ Found {len(result)} unique papers across all AI categories")
        return result

    def _fetch_category(self, category: str) -> list[Paper]:
        """Fetches papers for a single arXiv category."""

        # Build search query — filter by category + sort by submission date
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ArXiv Agent] ⚠️  Failed to fetch category {category}: {e}")
            return []

        return self._parse_response(response.text, category)

    def _parse_response(self, xml_text: str, category: str) -> list[Paper]:
        """
        Parses the Atom XML response from arXiv API.
        Filters to only papers published within our time window.
        """
        papers = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.hours_back)

        # ArXiv returns Atom XML — parse it
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", namespace)

        for entry in entries:
            try:
                # Extract published date
                published_str = entry.find("atom:published", namespace).text
                published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

                # Skip papers older than our cutoff
                if published_dt < cutoff_time:
                    continue

                # Extract paper ID (last part of the URL)
                paper_url = entry.find("atom:id", namespace).text.strip()
                paper_id = paper_url.split("/abs/")[-1]

                # Extract authors
                author_elements = entry.findall("atom:author", namespace)
                authors = [
                    a.find("atom:name", namespace).text
                    for a in author_elements
                ]

                # Extract categories
                cat_elements = entry.findall(
                    "{http://arxiv.org/schemas/atom}primary_category"
                )
                categories = [c.get("term", "") for c in cat_elements] or [category]

                paper = Paper(
                    paper_id=paper_id,
                    title=entry.find("atom:title", namespace).text.strip().replace("\n", " "),
                    authors=authors,
                    abstract=entry.find("atom:summary", namespace).text.strip().replace("\n", " "),
                    published_date=published_dt.strftime("%Y-%m-%d %H:%M UTC"),
                    url=paper_url,
                    source="arxiv",
                    categories=categories,
                )
                papers.append(paper)

            except Exception as e:
                # Skip malformed entries silently
                print(f"[ArXiv Agent] ⚠️  Skipping malformed entry: {e}")
                continue

        return papers


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = ArxivFetcherAgent(max_results=10, hours_back=48)
    papers = agent.fetch()

    print(f"\n{'='*60}")
    print(f"  📄 ARXIV PAPERS FETCHED: {len(papers)}")
    print(f"{'='*60}")

    for i, p in enumerate(papers[:3], 1):  # Preview first 3
        print(f"\n[{i}] {p.title}")
        print(f"    Authors : {', '.join(p.authors[:2])}{'...' if len(p.authors) > 2 else ''}")
        print(f"    Date    : {p.published_date}")
        print(f"    URL     : {p.url}")
        print(f"    Abstract: {p.abstract[:150]}...")
