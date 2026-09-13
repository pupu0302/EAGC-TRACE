"""
Document profiling and auditability feature extraction.

This module defines the DocAgg data structure and functions for building
doc-level profiles from canonical_v2 registries.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import re
from bs4 import BeautifulSoup


@dataclass
class DocAgg:
    """
    Aggregated document-level statistics and auditability proxies.
    
    This dataclass accumulates counts from evidence_registry, unit_registry,
    and table_registry to build a comprehensive doc profile.
    """
    # Basic identifiers
    doc_id: str
    ticker: str
    year: int
    version: str = ""
    
    # Structure: scale indicators
    n_sentences: int = 0
    n_units_text: int = 0
    n_units_table: int = 0
    n_tables: int = 0
    n_table_rows: int = 0
    n_table_cells: int = 0
    
    # Numeric grounding proxies (sentence-side)
    n_sentences_with_number: int = 0
    n_year_mentions: int = 0
    n_unit_mentions: int = 0
    
    # Standards/frameworks hits
    standards_hits: Dict[str, int] = field(default_factory=dict)
    
    # Methodology/boundary hits
    methodology_hits: Dict[str, int] = field(default_factory=dict)
    
    # Assurance hits
    assurance_hits: Dict[str, int] = field(default_factory=dict)
    
    # Table-side heuristics
    n_cells_numeric: int = 0
    n_cells_parsed_from_html: int = 0  # Actual cell count from HTML parsing
    n_tables_with_year_header: int = 0
    n_tables_with_unit_header: int = 0
    
    # Accumulator for table dimensions
    _table_row_counts: List[int] = field(default_factory=list)
    _table_cell_counts: List[int] = field(default_factory=list)


def init_doc_agg(doc_id: str, ticker: str = "", year: int = 0, version: str = "") -> DocAgg:
    """
    Initialize a new DocAgg for a document.
    
    Args:
        doc_id: Document identifier
        ticker: Company ticker
        year: Report year
        version: Version suffix (e.g., "_A", "_B")
        
    Returns:
        Initialized DocAgg instance
    """
    return DocAgg(
        doc_id=doc_id,
        ticker=ticker,
        year=year,
        version=version
    )


def update_from_sentence_evidence(agg: DocAgg, text: str, lex: Dict[str, Any]) -> None:
    """
    Update DocAgg from a sentence evidence record.
    
    This function extracts numeric signals, standards mentions, methodology signals,
    and assurance signals from sentence text.
    
    Args:
        agg: DocAgg to update (modified in place)
        text: Sentence text
        lex: Loaded lexicon dictionary
    """
    from .lexicon import (
        extract_numeric_signals,
        extract_standards_signals,
        extract_methodology_signals,
        extract_assurance_signals
    )
    
    agg.n_sentences += 1
    
    # Numeric grounding
    numeric_signals = extract_numeric_signals(text, lex)
    if numeric_signals['has_number']:
        agg.n_sentences_with_number += 1
    agg.n_year_mentions += numeric_signals['year_mentions']
    agg.n_unit_mentions += numeric_signals['unit_mentions']
    
    # Standards
    standards_signals = extract_standards_signals(text, lex)
    for std_name, count in standards_signals.items():
        if std_name not in agg.standards_hits:
            agg.standards_hits[std_name] = 0
        agg.standards_hits[std_name] += count
    
    # Methodology/boundary/restatement
    methodology_signals = extract_methodology_signals(text, lex)
    for key, count in methodology_signals.items():
        if key not in agg.methodology_hits:
            agg.methodology_hits[key] = 0
        agg.methodology_hits[key] += count
    
    # Assurance
    assurance_signals = extract_assurance_signals(text, lex)
    for key, count in assurance_signals.items():
        if key not in agg.assurance_hits:
            agg.assurance_hits[key] = 0
        agg.assurance_hits[key] += count


def update_from_unit_record(agg: DocAgg, unit_type: str) -> None:
    """
    Update DocAgg from a unit_registry record.
    
    Args:
        agg: DocAgg to update (modified in place)
        unit_type: Unit type string (e.g., "text_sentence", "table")
    """
    if unit_type == "table":
        agg.n_units_table += 1
    elif "text" in unit_type.lower() or "sentence" in unit_type.lower():
        agg.n_units_text += 1


def update_from_table_record(agg: DocAgg, table_rec: Dict[str, Any], lex: Dict[str, Any]) -> None:
    """
    Update DocAgg from a table_registry record.
    
    This function extracts:
    - Table structure: rows, cells
    - Table heuristics: year in header, unit in header, numeric cells
    
    Args:
        agg: DocAgg to update (modified in place)
        table_rec: Table registry record
        lex: Loaded lexicon dictionary
    """
    from .lexicon import is_numeric_cell, has_year_in_header, has_unit_in_header
    
    agg.n_tables += 1
    
    # Get table dimensions
    total_rows = table_rec.get('total_rows', 0)
    total_cols = table_rec.get('total_cols', 0)
    total_cells = total_rows * total_cols
    
    agg.n_table_rows += total_rows
    agg.n_table_cells += total_cells
    
    # Track for per-table stats
    agg._table_row_counts.append(total_rows)
    agg._table_cell_counts.append(total_cells)
    
    # Check headers for year/unit signals
    col_headers = table_rec.get('col_headers', [])
    if col_headers:
        header_text = ' '.join([str(h) for h in col_headers if h])
        if has_year_in_header(header_text, lex):
            agg.n_tables_with_year_header += 1
        if has_unit_in_header(header_text, lex):
            agg.n_tables_with_unit_header += 1
    
    # Count numeric cells (if table_html available)
    table_html = table_rec.get('table_html', '')
    if table_html:
        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            cells = soup.find_all('td')
            agg.n_cells_parsed_from_html += len(cells)  # Track actual parsed cells
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                if is_numeric_cell(cell_text, lex):
                    agg.n_cells_numeric += 1
        except Exception:
            # If parsing fails, skip numeric cell detection
            pass


def finalize_features(agg: DocAgg) -> Dict[str, Any]:
    """
    Compute derived features from DocAgg and return flat feature dictionary.
    
    This function:
    1. Computes normalized rates (per 1k sentences, per table averages)
    2. Aggregates boolean flags from counts
    3. Computes auditability component features (raw, not z-scored yet)
    
    Args:
        agg: DocAgg with accumulated counts
        
    Returns:
        Flat dictionary suitable for CSV output
    """
    features = {
        # Basic identifiers
        'doc_id': agg.doc_id,
        'ticker': agg.ticker,
        'year': agg.year,
        'version': agg.version,
        'ticker_year_key': f"{agg.ticker}_{agg.year}",
        
        # Raw counts: structure
        'n_sentences': agg.n_sentences,
        'n_units_text': agg.n_units_text,
        'n_units_table': agg.n_units_table,
        'n_tables': agg.n_tables,
        'n_table_rows': agg.n_table_rows,
        'n_table_cells': agg.n_table_cells,
    }
    
    # Derived: normalized densities
    sentences_safe = max(agg.n_sentences, 1)  # Avoid division by zero
    features['tables_per_1k_sentences'] = (agg.n_tables / sentences_safe) * 1000
    features['cells_per_1k_sentences'] = (agg.n_table_cells / sentences_safe) * 1000
    
    # Average sentences per text unit
    if agg.n_units_text > 0:
        features['avg_sentences_per_text_unit'] = agg.n_sentences / agg.n_units_text
    else:
        features['avg_sentences_per_text_unit'] = 0.0
    
    # Average rows/cells per table
    if agg.n_tables > 0:
        features['rows_per_table'] = agg.n_table_rows / agg.n_tables
        features['cells_per_table'] = agg.n_table_cells / agg.n_tables
    else:
        features['rows_per_table'] = 0.0
        features['cells_per_table'] = 0.0
    
    # Numeric grounding proxies
    features['n_sentences_with_number'] = agg.n_sentences_with_number
    features['pct_sentences_with_number'] = (agg.n_sentences_with_number / sentences_safe) * 100
    features['n_year_mentions'] = agg.n_year_mentions
    features['year_mentions_per_1k_sentences'] = (agg.n_year_mentions / sentences_safe) * 1000
    features['n_unit_mentions'] = agg.n_unit_mentions
    features['unit_mentions_per_1k_sentences'] = (agg.n_unit_mentions / sentences_safe) * 1000
    
    # Standards (count + bool)
    features['standards_count'] = sum(agg.standards_hits.values())
    for std_name in ['gri', 'sasb', 'tcfd', 'cdp', 'un_sdg', 'ghg_protocol']:
        count = agg.standards_hits.get(std_name, 0)
        features[f'has_{std_name}'] = int(count > 0)
    
    # Methodology/boundary/restatement
    features['methodology_mentions_count'] = agg.methodology_hits.get('has_methodology_section', 0)
    features['has_methodology_section'] = int(features['methodology_mentions_count'] > 0)
    
    scope_counts = (
        agg.methodology_hits.get('scope_1', 0) +
        agg.methodology_hits.get('scope_2', 0) +
        agg.methodology_hits.get('scope_3', 0)
    )
    features['scope_mentions_count'] = scope_counts
    features['has_scope_1'] = int(agg.methodology_hits.get('scope_1', 0) > 0)
    features['has_scope_2'] = int(agg.methodology_hits.get('scope_2', 0) > 0)
    features['has_scope_3'] = int(agg.methodology_hits.get('scope_3', 0) > 0)
    
    features['restatement_mentions_count'] = agg.methodology_hits.get('restatement', 0)
    features['has_restatement'] = int(features['restatement_mentions_count'] > 0)
    
    # Assurance
    features['assurance_mentions_count'] = agg.assurance_hits.get('assurance', 0)
    features['has_assurance'] = int(features['assurance_mentions_count'] > 0)
    features['has_limited_assurance'] = int(agg.assurance_hits.get('limited_assurance', 0) > 0)
    features['has_reasonable_assurance'] = int(agg.assurance_hits.get('reasonable_assurance', 0) > 0)
    
    assurance_standards_count = sum([
        agg.assurance_hits.get('isae_3000', 0),
        agg.assurance_hits.get('aa1000', 0),
        agg.assurance_hits.get('iso_14064', 0),
        agg.assurance_hits.get('iso_14001', 0),
        agg.assurance_hits.get('iso_45001', 0),
        agg.assurance_hits.get('sox', 0),
    ])
    features['assurance_named_standards_count'] = assurance_standards_count
    
    # Table heuristics
    if agg.n_cells_parsed_from_html > 0:
        # Use actual parsed cell count if available (more accurate)
        features['pct_cells_numeric'] = (agg.n_cells_numeric / agg.n_cells_parsed_from_html) * 100
    elif agg.n_table_cells > 0:
        # Fallback to computed cell count
        features['pct_cells_numeric'] = (agg.n_cells_numeric / agg.n_table_cells) * 100
    else:
        features['pct_cells_numeric'] = 0.0
    
    if agg.n_tables > 0:
        features['pct_tables_with_year_header'] = (agg.n_tables_with_year_header / agg.n_tables) * 100
        features['pct_tables_with_unit_header'] = (agg.n_tables_with_unit_header / agg.n_tables) * 100
    else:
        features['pct_tables_with_year_header'] = 0.0
        features['pct_tables_with_unit_header'] = 0.0
    
    # Raw auditability component features (not z-scored yet)
    # These will be z-scored in the main script across all docs
    features['audit_textual_def_raw'] = (
        features['standards_count'] +
        features['scope_mentions_count'] +
        features['methodology_mentions_count'] +
        features['restatement_mentions_count']
    )
    
    features['audit_assurance_raw'] = (
        features['assurance_mentions_count'] +
        features['assurance_named_standards_count']
    )
    
    features['audit_numeric_raw'] = (
        features['pct_sentences_with_number'] +
        features['unit_mentions_per_1k_sentences'] +
        features['pct_cells_numeric']
    )
    
    features['audit_table_structure_raw'] = (
        features['tables_per_1k_sentences'] +
        features['cells_per_table']
    )
    
    return features


def select_canonical_doc(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select canonical document from multiple versions (same ticker-year).
    
    Selection criteria (in order):
    1. Max n_sentences
    2. Max n_table_cells (tie-break)
    3. Lexicographic min doc_id (final tie-break)
    
    Args:
        profiles: List of doc profiles with same ticker-year
        
    Returns:
        Selected canonical profile
    """
    if not profiles:
        return None
    
    if len(profiles) == 1:
        return profiles[0]
    
    # Sort by: (-n_sentences, -n_table_cells, doc_id)
    sorted_profiles = sorted(
        profiles,
        key=lambda p: (-p['n_sentences'], -p['n_table_cells'], p['doc_id'])
    )
    
    return sorted_profiles[0]
