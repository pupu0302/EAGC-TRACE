"""
I/O streaming utilities for reading JSONL registries.

This module provides efficient streaming readers for the canonical_v2 registries.
"""

import json
from typing import Iterator, Dict, Any, Optional, Set
from pathlib import Path


def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL file line by line, yielding parsed JSON objects.
    
    Args:
        path: Path to JSONL file
        
    Yields:
        Parsed JSON dictionary for each line
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {path}: {e}")
                    continue


def load_blacklist(path: Optional[str]) -> Set[str]:
    """
    Load document blacklist from a text file.
    
    Args:
        path: Path to blacklist file (one doc_id per line), or None
        
    Returns:
        Set of blacklisted doc_ids
    """
    if not path or not Path(path).exists():
        return set()
    
    blacklist = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            doc_id = line.strip()
            if doc_id:
                blacklist.add(doc_id)
    
    return blacklist


def parse_doc_id(doc_id: str) -> Dict[str, Any]:
    """
    Parse doc_id into components (ticker, year, version).
    
    Expected formats:
        - TICKER_YEAR (e.g., "AAPL_2022")
        - TICKER_YEAR_VERSION (e.g., "AAPL_2022_A")
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Dictionary with 'ticker', 'year', 'version'
    """
    parts = doc_id.split('_')
    
    result = {
        'ticker': '',
        'year': None,
        'version': ''
    }
    
    if len(parts) >= 2:
        result['ticker'] = parts[0]
        try:
            result['year'] = int(parts[1])
        except (ValueError, IndexError):
            result['year'] = None
        
        # Version is optional (3rd part)
        if len(parts) >= 3:
            result['version'] = parts[2]
    
    return result


def make_ticker_year_key(ticker: str, year: int) -> str:
    """
    Create canonical ticker-year key for grouping.
    
    Args:
        ticker: Company ticker
        year: Report year
        
    Returns:
        String key in format "TICKER_YEAR"
    """
    return f"{ticker}_{year}"
