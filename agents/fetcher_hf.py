"""
AGENT: HuggingFace Papers Fetcher
PURPOSE: Scrapes the latest AI papers from huggingface.co/papers
         HF Papers is community-curated — high signal, trending papers only.
         No official API exists, so we scrape the HTML page.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from dataclasses import dataclass

# Import our shared Paper model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.fetcher_arxiv import Paper


# ── HuggingFace Papers Fetcher Agent ─────────────────────────────────────────

class HuggingFaceFetcherAgent:
    """
    Scrapes trending/latest papers from https://huggingface.co/papers
    
    Why HF Papers?
    - Community-curated: only notable papers make it here
    - High signal-to-noise ratio vs raw arXiv
    - Updated daily with upvotes showing community interest
    
    Scraping approach:
    - HF Papers renders server-side HTML — straightforward to parse
    - We extract paper cards from the main listing page
    - Then optionally fetch each paper page for full abstract
    """

    HF_PAPERS_URL = "https://huggingface.co/papers"

    HEADERS = {
        # Mimic a real browser to avoid being blocked
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, max_results: int = 20):
        """
        Args:
            max_results: Max number of papers to return from HF Papers
        """
        self.max_results = max_results

    def fetch(self) -> list[Paper]:
        """
        Main entry point — scrapes HF Papers and returns Paper objects.
        """
        print(f"[HF Agent] Fetching papers from huggingface.co/papers...")

        html = self._get_page(self.HF_PAPERS_URL)
        if not html:
            return []

        papers = self._parse_papers(html)
        print(f"[HF Agent] ✅ Found {len(papers)} papers from HuggingFace")
        return papers

    def _get_page(self, url: str) -> str | None:
        """Fetches raw HTML from a URL with error handling."""
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"[HF Agent] ⚠️  Failed to fetch {url}: {e}")
            return None

    def _parse_papers(self, html: str) -> list[Paper]:
        """
        Parses the HuggingFace Papers listing page.
        
        HF Papers page structure (as of 2024-2025):
        - Each paper is in an <article> tag
        - Title is in an <h3> tag inside the article
        - Paper link points to /papers/{arxiv_id}
        - We extract what we can from the listing, 
          then fetch individual pages for abstracts
        """
        soup = BeautifulSoup(html, "html.parser")
        papers = []

        # Find all paper article cards on the page
        # HF uses <article> tags for each paper card
        article_cards = soup.find_all("article", limit=self.max_results)

        if not article_cards:
            print("[HF Agent] ⚠️  No article cards found — page structure may have changed")
            # Fallback: try finding paper links directly
            return self._fallback_parse(soup)

        for card in article_cards:
            try:
                paper = self._parse_card(card)
                if paper:
                    papers.append(paper)
            except Exception as e:
                print(f"[HF Agent] ⚠️  Skipping card parse error: {e}")
                continue

        # If we got papers but no abstracts, enrich them
        papers_with_abstracts = []
        for paper in papers[:self.max_results]:
            enriched = self._enrich_with_abstract(paper)
            papers_with_abstracts.append(enriched)

        return papers_with_abstracts

    def _parse_card(self, card) -> Paper | None:
        """Extracts paper info from a single HF paper card."""

        # Find the title and link
        title_tag = card.find("h3") or card.find("h2")
        if not title_tag:
            return None

        title = title_tag.get_text(strip=True)
        if not title:
            return None

        # Find paper URL — links to /papers/{arxiv_id}
        link_tag = card.find("a", href=True)
        if not link_tag:
            return None

        href = link_tag["href"]
        # Handle relative URLs
        if href.startswith("/papers/"):
            paper_url = f"https://huggingface.co{href}"
            # Extract arxiv ID from URL like /papers/2401.12345
            arxiv_id = href.replace("/papers/", "").strip()
        elif "arxiv.org" in href:
            paper_url = href
            arxiv_id = href.split("/")[-1]
        else:
            return None

        # Try to find upvote count (signal of community interest)
        upvotes = 0
        upvote_tag = card.find(attrs={"data-target": True}) or card.find(class_=lambda c: c and "upvote" in c.lower() if c else False)
        if upvote_tag:
            try:
                upvotes = int(upvote_tag.get_text(strip=True))
            except (ValueError, AttributeError):
                pass

        return Paper(
            paper_id=f"hf_{arxiv_id}",
            title=title,
            authors=[],           # Will be filled by _enrich_with_abstract
            abstract="",          # Will be filled by _enrich_with_abstract
            published_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            url=paper_url,
            source="huggingface",
            categories=["AI/ML"], # HF Papers are all AI-related
        )

    def _enrich_with_abstract(self, paper: Paper) -> Paper:
        """
        Fetches the individual paper page on HF to get:
        - Full abstract
        - Authors
        - Published date
        
        HF paper pages (huggingface.co/papers/{id}) embed the arXiv metadata.
        """
        try:
            html = self._get_page(paper.url)
            if not html:
                return paper

            soup = BeautifulSoup(html, "html.parser")

            # Abstract — usually in a <p> with specific class or inside a section
            abstract = ""
            # Try multiple selectors as HF may update their layout
            abstract_candidates = [
                soup.find("div", class_=lambda c: c and "abstract" in c.lower() if c else False),
                soup.find("p", class_=lambda c: c and "abstract" in c.lower() if c else False),
                soup.find(attrs={"data-target": "abstract"}),
            ]
            for candidate in abstract_candidates:
                if candidate:
                    abstract = candidate.get_text(strip=True)
                    break

            # Fallback: grab the largest <p> tag (often the abstract)
            if not abstract:
                all_paragraphs = soup.find_all("p")
                if all_paragraphs:
                    abstract = max(all_paragraphs, key=lambda p: len(p.get_text())).get_text(strip=True)

            # Authors — look for author metadata
            authors = []
            author_section = soup.find(class_=lambda c: c and "author" in c.lower() if c else False)
            if author_section:
                author_tags = author_section.find_all("span") or author_section.find_all("a")
                authors = [a.get_text(strip=True) for a in author_tags if a.get_text(strip=True)]

            # Update the paper with enriched data
            paper.abstract = abstract[:1500] if abstract else "Abstract not available."
            paper.authors = authors if authors else ["See paper page"]
            return paper

        except Exception as e:
            print(f"[HF Agent] ⚠️  Could not enrich {paper.url}: {e}")
            return paper  # Return as-is if enrichment fails

    def _fallback_parse(self, soup: BeautifulSoup) -> list[Paper]:
        """
        Fallback parser if main article card parsing fails.
        Looks for any links pointing to /papers/ pattern.
        """
        print("[HF Agent] Trying fallback parser...")
        papers = []
        seen_ids = set()

        links = soup.find_all("a", href=lambda h: h and "/papers/" in h)
        for link in links[:self.max_results]:
            href = link["href"]
            paper_id = href.split("/papers/")[-1].strip("/")

            if paper_id in seen_ids or not paper_id:
                continue
            seen_ids.add(paper_id)

            title = link.get_text(strip=True)
            if len(title) < 10:  # Skip nav links etc
                continue

            paper = Paper(
                paper_id=f"hf_{paper_id}",
                title=title,
                authors=["See paper page"],
                abstract="",
                published_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                url=f"https://huggingface.co/papers/{paper_id}",
                source="huggingface",
                categories=["AI/ML"],
            )
            papers.append(self._enrich_with_abstract(paper))

        return papers


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = HuggingFaceFetcherAgent(max_results=5)
    papers = agent.fetch()

    print(f"\n{'='*60}")
    print(f"  🤗 HUGGINGFACE PAPERS FETCHED: {len(papers)}")
    print(f"{'='*60}")

    for i, p in enumerate(papers[:3], 1):
        print(f"\n[{i}] {p.title}")
        print(f"    Authors : {', '.join(p.authors[:2]) if p.authors else 'N/A'}")
        print(f"    URL     : {p.url}")
        print(f"    Abstract: {p.abstract[:150]}...")
