"""
Lexicon loader for auditability keywords and patterns.

This module loads and provides access to the auditability lexicon
defined in configs/auditability_lexicon.yaml.
"""

import yaml
import re
from typing import Dict, List, Any
from pathlib import Path


def load_lexicon(path: str) -> Dict[str, Any]:
    """
    Load the auditability lexicon from a YAML file.
    
    Args:
        path: Path to the lexicon YAML file
        
    Returns:
        Dictionary containing lexicon configuration with compiled regexes
        
    Example structure:
        {
            'numeric': {'number_regex': compiled_re, 'year_regex': compiled_re, ...},
            'standards': {...},
            'methodology': {...},
            'assurance': {...},
            'tables': {...},
            'auditability_index': {...}
        }
    """
    with open(path, 'r', encoding='utf-8') as f:
        lex = yaml.safe_load(f)
    
    # Pre-compile regex patterns for efficiency
    if 'numeric' in lex:
        if 'number_regex' in lex['numeric']:
            lex['numeric']['number_regex_compiled'] = re.compile(
                lex['numeric']['number_regex'], 
                re.IGNORECASE
            )
        if 'year_regex' in lex['numeric']:
            lex['numeric']['year_regex_compiled'] = re.compile(
                lex['numeric']['year_regex']
            )
    
    if 'tables' in lex:
        if 'header_year_regex' in lex['tables']:
            lex['tables']['header_year_regex_compiled'] = re.compile(
                lex['tables']['header_year_regex']
            )
        if 'cell_numeric_regex' in lex['tables']:
            lex['tables']['cell_numeric_regex_compiled'] = re.compile(
                lex['tables']['cell_numeric_regex']
            )
    
    return lex


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """
    Count occurrences of keywords in text (case-insensitive).
    
    Args:
        text: Text to search
        keywords: List of keyword strings/phrases
        
    Returns:
        Total count of keyword matches
    """
    if not text:
        return 0
    
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        kw_lower = kw.lower()
        count += text_lower.count(kw_lower)
    return count


def has_keyword(text: str, keywords: List[str]) -> bool:
    """
    Check if any keyword appears in text (case-insensitive).
    
    Args:
        text: Text to search
        keywords: List of keyword strings/phrases
        
    Returns:
        True if at least one keyword is found
    """
    return count_keyword_matches(text, keywords) > 0


def extract_numeric_signals(text: str, lex: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract numeric grounding signals from text.
    
    Args:
        text: Text to analyze
        lex: Loaded lexicon dictionary
        
    Returns:
        Dictionary with:
            - has_number: bool (presence of numeric patterns)
            - year_mentions: count of year mentions (19xx/20xx)
            - unit_mentions: count of ESG unit token mentions
    """
    results = {
        'has_number': False,
        'year_mentions': 0,
        'unit_mentions': 0
    }
    
    if not text or 'numeric' not in lex:
        return results
    
    numeric_lex = lex['numeric']
    
    # Check for numeric patterns
    if 'number_regex_compiled' in numeric_lex:
        matches = numeric_lex['number_regex_compiled'].findall(text)
        results['has_number'] = len(matches) > 0
    
    # Count year mentions
    if 'year_regex_compiled' in numeric_lex:
        matches = numeric_lex['year_regex_compiled'].findall(text)
        results['year_mentions'] = len(matches)
    
    # Count unit token mentions
    if 'unit_tokens' in numeric_lex:
        results['unit_mentions'] = count_keyword_matches(text, numeric_lex['unit_tokens'])
    
    return results


def extract_standards_signals(text: str, lex: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract standards/framework mentions from text.
    
    Args:
        text: Text to analyze
        lex: Loaded lexicon dictionary
        
    Returns:
        Dictionary with counts for each standard (gri, sasb, tcfd, cdp, un_sdg, ghg_protocol)
    """
    results = {}
    
    if not text or 'standards' not in lex:
        return results
    
    standards_lex = lex['standards']
    
    for std_name, std_config in standards_lex.items():
        if 'keywords' in std_config:
            results[std_name] = count_keyword_matches(text, std_config['keywords'])
        else:
            results[std_name] = 0
    
    return results


def extract_methodology_signals(text: str, lex: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract methodology/boundary/restatement signals from text.
    
    Args:
        text: Text to analyze
        lex: Loaded lexicon dictionary
        
    Returns:
        Dictionary with counts for methodology indicators
    """
    results = {}
    
    if not text or 'methodology' not in lex:
        return results
    
    methodology_lex = lex['methodology']
    
    for key, config in methodology_lex.items():
        if 'keywords' in config:
            results[key] = count_keyword_matches(text, config['keywords'])
        else:
            results[key] = 0
    
    return results


def extract_assurance_signals(text: str, lex: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract assurance/verification signals from text.
    
    Args:
        text: Text to analyze
        lex: Loaded lexicon dictionary
        
    Returns:
        Dictionary with counts for assurance indicators
    """
    results = {}
    
    if not text or 'assurance' not in lex:
        return results
    
    assurance_lex = lex['assurance']
    
    for key, config in assurance_lex.items():
        if 'keywords' in config:
            results[key] = count_keyword_matches(text, config['keywords'])
        else:
            results[key] = 0
    
    return results


def is_numeric_cell(cell_text: str, lex: Dict[str, Any]) -> bool:
    """
    Check if table cell contains primarily numeric content.
    
    Args:
        cell_text: Cell text content
        lex: Loaded lexicon dictionary
        
    Returns:
        True if cell matches numeric pattern
    """
    if not cell_text or 'tables' not in lex:
        return False
    
    tables_lex = lex['tables']
    
    if 'cell_numeric_regex_compiled' in tables_lex:
        # Clean whitespace and check
        cleaned = cell_text.strip()
        return tables_lex['cell_numeric_regex_compiled'].match(cleaned) is not None
    
    return False


def has_year_in_header(header_text: str, lex: Dict[str, Any]) -> bool:
    """
    Check if table header contains year mentions.
    
    Args:
        header_text: Header text content
        lex: Loaded lexicon dictionary
        
    Returns:
        True if year pattern found in header
    """
    if not header_text or 'tables' not in lex:
        return False
    
    tables_lex = lex['tables']
    
    if 'header_year_regex_compiled' in tables_lex:
        return tables_lex['header_year_regex_compiled'].search(header_text) is not None
    
    return False


def has_unit_in_header(header_text: str, lex: Dict[str, Any]) -> bool:
    """
    Check if table header contains unit tokens.
    
    Args:
        header_text: Header text content
        lex: Loaded lexicon dictionary
        
    Returns:
        True if unit tokens found in header
    """
    if not header_text or 'tables' not in lex:
        return False
    
    tables_lex = lex['tables']
    
    if 'header_unit_tokens' in tables_lex:
        return has_keyword(header_text, tables_lex['header_unit_tokens'])
    
    return False
