# agents/__init__.py
# Makes the agents/ directory a Python package.
# Exposes all agent classes and shared data models for clean imports.
# Usage elsewhere: from agents import ArxivFetcherAgent, Paper etc.

from agents.fetcher_arxiv    import ArxivFetcherAgent, Paper
from agents.fetcher_hf       import HuggingFaceFetcherAgent
from agents.filter_agent     import FilterRankAgent
from agents.summariser_agent import SummariserAgent, SummarisedPaper
from agents.newsletter_agent import NewsletterAgent
