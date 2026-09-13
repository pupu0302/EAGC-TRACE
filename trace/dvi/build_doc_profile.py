#!/usr/bin/env python3
"""
Build Document Profile for ESG Accountability Benchmark

This script generates doc-level profiling and auditability proxies from
canonical_v2 registries for:
- Report-level DVI analysis
- External-alignment Spearman correlation analysis
- Corpus profiling and robustness analysis

Author: ESG Accountability Benchmark Team
Version: 1.0.0
Date: 2026-01-02
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from dvi_core.io_stream import stream_jsonl, load_blacklist, parse_doc_id, make_ticker_year_key
from dvi_core.lexicon import load_lexicon
from dvi_core.profile import (
    init_doc_agg, 
    update_from_sentence_evidence,
    update_from_unit_record,
    update_from_table_record,
    finalize_features,
    select_canonical_doc
)


def build_doc_profiles(
    evidence_path: str,
    units_path: str,
    tables_path: str,
    lexicon_path: str,
    exclude_docs: set
) -> List[Dict[str, Any]]:
    """
    Build document profiles from canonical_v2 registries.
    
    Args:
        evidence_path: Path to evidence_registry.jsonl
        units_path: Path to unit_registry.jsonl
        tables_path: Path to table_registry.jsonl
        lexicon_path: Path to auditability_lexicon.yaml
        exclude_docs: Set of doc_ids to exclude
        
    Returns:
        List of doc profile dictionaries
    """
    print("Loading lexicon...")
    lex = load_lexicon(lexicon_path)
    
    print("Initializing document aggregators...")
    doc_aggs: Dict[str, Any] = {}
    
    # Phase 1: Process evidence_registry for sentence counts and text signals
    print(f"Processing evidence_registry: {evidence_path}")
    sentence_count = 0
    cell_count = 0
    
    for record in stream_jsonl(evidence_path):
        doc_id = record.get('doc_id')
        if not doc_id or doc_id in exclude_docs:
            continue
        
        # Initialize doc_agg if first time seeing this doc
        if doc_id not in doc_aggs:
            parsed = parse_doc_id(doc_id)
            doc_aggs[doc_id] = init_doc_agg(
                doc_id=doc_id,
                ticker=record.get('ticker', parsed['ticker']),
                year=record.get('year', parsed['year']),
                version=parsed['version']
            )
        
        # Process sentence evidence (skip table_cell for now)
        evid_type = record.get('type', '')
        text = record.get('text', '')
        
        if evid_type == 'sentence' and text:
            update_from_sentence_evidence(doc_aggs[doc_id], text, lex)
            sentence_count += 1
        elif evid_type == 'table_cell':
            cell_count += 1
    
    print(f"  Processed {sentence_count} sentences, {cell_count} table cells")
    print(f"  Found {len(doc_aggs)} unique documents")
    
    # Phase 2: Process unit_registry for unit counts
    print(f"Processing unit_registry: {units_path}")
    unit_count = 0
    
    for record in stream_jsonl(units_path):
        doc_id = record.get('doc_id')
        if not doc_id or doc_id in exclude_docs:
            continue
        
        if doc_id in doc_aggs:
            unit_type = record.get('unit_type', '')
            update_from_unit_record(doc_aggs[doc_id], unit_type)
            unit_count += 1
    
    print(f"  Processed {unit_count} units")
    
    # Phase 3: Process table_registry for table structure and heuristics
    print(f"Processing table_registry: {tables_path}")
    table_count = 0
    
    for record in stream_jsonl(tables_path):
        doc_id = record.get('doc_id')
        if not doc_id or doc_id in exclude_docs:
            continue
        
        if doc_id in doc_aggs:
            update_from_table_record(doc_aggs[doc_id], record, lex)
            table_count += 1
    
    print(f"  Processed {table_count} tables")
    
    # Phase 4: Finalize features
    print("Finalizing features...")
    profiles = []
    for doc_id, agg in doc_aggs.items():
        features = finalize_features(agg)
        profiles.append(features)
    
    print(f"  Generated {len(profiles)} doc profiles")
    
    return profiles


def compute_z_scores(profiles: List[Dict[str, Any]], feature_names: List[str]) -> None:
    """
    Compute z-scores for specified features across all profiles.
    
    Modifies profiles in-place, adding z-scored versions of features.
    
    Args:
        profiles: List of doc profiles
        feature_names: List of feature names to z-score
    """
    for feat_name in feature_names:
        values = [p.get(feat_name, 0) for p in profiles]
        values_array = np.array(values, dtype=float)
        
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        if std_val > 0:
            z_scores = (values_array - mean_val) / std_val
        else:
            z_scores = np.zeros_like(values_array)
        
        # Add z-scored feature
        z_feat_name = feat_name.replace('_raw', '_z')
        for i, profile in enumerate(profiles):
            profile[z_feat_name] = float(z_scores[i])


def compute_auditability_index(profiles: List[Dict[str, Any]]) -> None:
    """
    Compute auditability index from z-scored component features.
    
    Modifies profiles in-place, adding 'auditability_index'.
    
    Args:
        profiles: List of doc profiles (must have z-scored components)
    """
    for profile in profiles:
        components = [
            profile.get('audit_textual_def_z', 0),
            profile.get('audit_assurance_z', 0),
            profile.get('audit_numeric_z', 0),
            profile.get('audit_table_structure_z', 0)
        ]
        profile['auditability_index'] = float(np.mean(components))


def mark_canonical_docs(profiles: List[Dict[str, Any]]) -> None:
    """
    Mark canonical document for each ticker-year (for external validity analysis).
    
    Modifies profiles in-place, adding 'is_canonical_ticker_year' boolean.
    
    Args:
        profiles: List of doc profiles
    """
    # Group by ticker-year
    ticker_year_groups = defaultdict(list)
    for profile in profiles:
        key = profile['ticker_year_key']
        ticker_year_groups[key].append(profile)
    
    # Select canonical for each group
    for key, group in ticker_year_groups.items():
        canonical = select_canonical_doc(group)
        canonical_doc_id = canonical['doc_id']
        
        # Mark all profiles
        for profile in group:
            profile['is_canonical_ticker_year'] = int(profile['doc_id'] == canonical_doc_id)


def generate_sanity_report(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate sanity check report for doc profiles.
    
    Args:
        profiles: List of doc profiles
        
    Returns:
        Dictionary with sanity check statistics
    """
    df = pd.DataFrame(profiles)
    
    report = {
        'total_docs': len(profiles),
        'unique_tickers': df['ticker'].nunique(),
        'unique_years': df['year'].nunique(),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
    }
    
    # Missing data checks
    report['docs_with_no_sentences'] = int((df['n_sentences'] == 0).sum())
    report['docs_with_no_tables'] = int((df['n_tables'] == 0).sum())
    report['docs_with_no_text_units'] = int((df['n_units_text'] == 0).sum())
    
    # Distribution statistics
    key_features = [
        'n_sentences', 'n_tables', 'n_table_cells',
        'pct_sentences_with_number', 'standards_count',
        'assurance_mentions_count', 'auditability_index'
    ]
    
    report['distributions'] = {}
    for feat in key_features:
        if feat in df.columns:
            report['distributions'][feat] = {
                'min': float(df[feat].min()),
                'p25': float(df[feat].quantile(0.25)),
                'p50': float(df[feat].quantile(0.50)),
                'p75': float(df[feat].quantile(0.75)),
                'p90': float(df[feat].quantile(0.90)),
                'max': float(df[feat].max()),
                'mean': float(df[feat].mean()),
                'std': float(df[feat].std())
            }
    
    # Extreme outliers
    report['extreme_outliers'] = {}
    
    # Top 5 by cells
    top_cells = df.nlargest(5, 'n_table_cells')[['doc_id', 'n_table_cells']].to_dict('records')
    report['extreme_outliers']['top_5_by_cells'] = top_cells
    
    # Top 5 by auditability
    top_audit = df.nlargest(5, 'auditability_index')[['doc_id', 'auditability_index']].to_dict('records')
    report['extreme_outliers']['top_5_by_auditability'] = top_audit
    
    # Bottom 5 by auditability
    bottom_audit = df.nsmallest(5, 'auditability_index')[['doc_id', 'auditability_index']].to_dict('records')
    report['extreme_outliers']['bottom_5_by_auditability'] = bottom_audit
    
    return report


def generate_schema_doc(lexicon_path: str) -> Dict[str, Any]:
    """
    Generate schema documentation for doc_profile.csv.
    
    Args:
        lexicon_path: Path to lexicon file
        
    Returns:
        Schema dictionary describing all fields
    """
    schema = {
        'version': '1.0.0',
        'description': 'Document-level profiling and auditability features for ESG reports',
        'source': 'canonical_v2 registries (evidence, unit, table)',
        'lexicon': lexicon_path,
        'fields': {}
    }
    
    # Define all fields (for paper writing)
    fields = schema['fields']
    
    # Identifiers
    fields['doc_id'] = {'type': 'string', 'description': 'Unique document identifier (may include version suffix)'}
    fields['ticker'] = {'type': 'string', 'description': 'Company stock ticker'}
    fields['year'] = {'type': 'int', 'description': 'Report year'}
    fields['version'] = {'type': 'string', 'description': 'Version suffix (e.g., _A, _B) if multiple reports exist'}
    fields['ticker_year_key'] = {'type': 'string', 'description': 'Canonical ticker_year key for joining'}
    fields['is_canonical_ticker_year'] = {'type': 'int', 'description': 'Boolean: 1 if selected as canonical for this ticker-year'}
    
    # Structure
    fields['n_sentences'] = {'type': 'int', 'description': 'Total sentence count from evidence_registry'}
    fields['n_units_text'] = {'type': 'int', 'description': 'Count of text-based units'}
    fields['n_units_table'] = {'type': 'int', 'description': 'Count of table units'}
    fields['n_tables'] = {'type': 'int', 'description': 'Total table count'}
    fields['n_table_rows'] = {'type': 'int', 'description': 'Total rows across all tables'}
    fields['n_table_cells'] = {'type': 'int', 'description': 'Total cells across all tables'}
    
    # Normalized densities
    fields['tables_per_1k_sentences'] = {'type': 'float', 'description': 'Tables per 1000 sentences'}
    fields['cells_per_1k_sentences'] = {'type': 'float', 'description': 'Table cells per 1000 sentences'}
    fields['avg_sentences_per_text_unit'] = {'type': 'float', 'description': 'Average sentences per text unit (~10 validates chunking)'}
    fields['rows_per_table'] = {'type': 'float', 'description': 'Average rows per table'}
    fields['cells_per_table'] = {'type': 'float', 'description': 'Average cells per table'}
    
    # Numeric grounding
    fields['n_sentences_with_number'] = {'type': 'int', 'description': 'Sentences containing numeric patterns'}
    fields['pct_sentences_with_number'] = {'type': 'float', 'description': 'Percentage of sentences with numbers'}
    fields['n_year_mentions'] = {'type': 'int', 'description': 'Count of year mentions (19xx/20xx)'}
    fields['year_mentions_per_1k_sentences'] = {'type': 'float', 'description': 'Year mentions per 1000 sentences'}
    fields['n_unit_mentions'] = {'type': 'int', 'description': 'Count of ESG unit token mentions'}
    fields['unit_mentions_per_1k_sentences'] = {'type': 'float', 'description': 'Unit mentions per 1000 sentences'}
    
    # Standards/frameworks
    fields['standards_count'] = {'type': 'int', 'description': 'Total count of standards/framework mentions'}
    for std in ['gri', 'sasb', 'tcfd', 'cdp', 'un_sdg', 'ghg_protocol']:
        fields[f'has_{std}'] = {'type': 'int', 'description': f'Boolean: 1 if {std.upper()} mentioned'}
    
    # Methodology/boundary
    fields['methodology_mentions_count'] = {'type': 'int', 'description': 'Count of methodology/boundary keywords'}
    fields['has_methodology_section'] = {'type': 'int', 'description': 'Boolean: 1 if methodology indicators found'}
    fields['scope_mentions_count'] = {'type': 'int', 'description': 'Total scope 1/2/3 mentions'}
    fields['has_scope_1'] = {'type': 'int', 'description': 'Boolean: 1 if scope 1 mentioned'}
    fields['has_scope_2'] = {'type': 'int', 'description': 'Boolean: 1 if scope 2 mentioned'}
    fields['has_scope_3'] = {'type': 'int', 'description': 'Boolean: 1 if scope 3 mentioned'}
    fields['restatement_mentions_count'] = {'type': 'int', 'description': 'Count of restatement/revision keywords'}
    fields['has_restatement'] = {'type': 'int', 'description': 'Boolean: 1 if restatement mentioned'}
    
    # Assurance
    fields['assurance_mentions_count'] = {'type': 'int', 'description': 'Count of assurance keywords'}
    fields['has_assurance'] = {'type': 'int', 'description': 'Boolean: 1 if assurance mentioned'}
    fields['has_limited_assurance'] = {'type': 'int', 'description': 'Boolean: 1 if limited assurance mentioned'}
    fields['has_reasonable_assurance'] = {'type': 'int', 'description': 'Boolean: 1 if reasonable assurance mentioned'}
    fields['assurance_named_standards_count'] = {'type': 'int', 'description': 'Count of named assurance standards (ISAE, AA1000, ISO)'}
    
    # Table heuristics
    fields['pct_cells_numeric'] = {'type': 'float', 'description': 'Percentage of table cells that are numeric'}
    fields['pct_tables_with_year_header'] = {'type': 'float', 'description': 'Percentage of tables with year in header'}
    fields['pct_tables_with_unit_header'] = {'type': 'float', 'description': 'Percentage of tables with unit tokens in header'}
    
    # Auditability components (z-scored)
    fields['audit_textual_def_z'] = {'type': 'float', 'description': 'Z-score of textual definition component'}
    fields['audit_assurance_z'] = {'type': 'float', 'description': 'Z-score of assurance component'}
    fields['audit_numeric_z'] = {'type': 'float', 'description': 'Z-score of numeric grounding component'}
    fields['audit_table_structure_z'] = {'type': 'float', 'description': 'Z-score of table structure component'}
    fields['auditability_index'] = {'type': 'float', 'description': 'Mean of z-scored auditability components'}
    
    return schema


def main():
    parser = argparse.ArgumentParser(
        description='Build Document Profile for ESG Accountability Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python trace/dvi/build_doc_profile.py \\
    --evidence data/canonical_v2/evidence_registry.jsonl \\
    --units data/canonical_v2/unit_registry.jsonl \\
    --tables data/canonical_v2/table_registry.jsonl \\
    --exclude_docs data/canonical_v2/blacklist_docs.txt \\
    --lexicon configs/auditability_lexicon.yaml \\
    --out outputs/dvi
        """
    )
    
    parser.add_argument(
        '--evidence',
        required=True,
        help='Path to evidence_registry.jsonl'
    )
    parser.add_argument(
        '--units',
        required=True,
        help='Path to unit_registry.jsonl'
    )
    parser.add_argument(
        '--tables',
        required=True,
        help='Path to table_registry.jsonl'
    )
    parser.add_argument(
        '--exclude_docs',
        default=None,
        help='Path to blacklist_docs.txt (one doc_id per line)'
    )
    parser.add_argument(
        '--lexicon',
        required=True,
        help='Path to auditability_lexicon.yaml'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load blacklist
    print("Loading blacklist...")
    exclude_docs = load_blacklist(args.exclude_docs)
    print(f"  Excluding {len(exclude_docs)} documents: {exclude_docs}")
    
    # Build profiles
    profiles = build_doc_profiles(
        evidence_path=args.evidence,
        units_path=args.units,
        tables_path=args.tables,
        lexicon_path=args.lexicon,
        exclude_docs=exclude_docs
    )
    
    if not profiles:
        print("ERROR: No profiles generated!")
        return 1
    
    # Compute z-scores for auditability components
    print("Computing z-scores for auditability components...")
    z_score_features = [
        'audit_textual_def_raw',
        'audit_assurance_raw',
        'audit_numeric_raw',
        'audit_table_structure_raw'
    ]
    compute_z_scores(profiles, z_score_features)
    
    # Compute auditability index
    print("Computing auditability index...")
    compute_auditability_index(profiles)
    
    # Mark canonical docs
    print("Marking canonical documents for external validity...")
    mark_canonical_docs(profiles)
    
    # Generate sanity report
    print("Generating sanity check report...")
    sanity_report = generate_sanity_report(profiles)
    
    # Generate schema documentation
    print("Generating schema documentation...")
    schema = generate_schema_doc(args.lexicon)
    
    # Write outputs
    print("Writing outputs...")
    
    # 1. doc_profile.csv
    df = pd.DataFrame(profiles)
    # Order columns logically
    id_cols = ['doc_id', 'ticker', 'year', 'version', 'ticker_year_key', 'is_canonical_ticker_year']
    structure_cols = [c for c in df.columns if c.startswith('n_') and c in df.columns]
    rate_cols = [c for c in df.columns if ('_per_' in c or 'pct_' in c or 'avg_' in c) and 'audit' not in c]
    has_cols = [c for c in df.columns if c.startswith('has_')]
    count_cols = [c for c in df.columns if c.endswith('_count') and 'n_' not in c]
    audit_cols = [c for c in df.columns if 'audit' in c]
    
    ordered_cols = id_cols + structure_cols + rate_cols + count_cols + has_cols + audit_cols
    # Handle any remaining columns
    remaining = [c for c in df.columns if c not in ordered_cols]
    ordered_cols = ordered_cols + remaining
    # Filter to only existing columns
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    
    df = df[ordered_cols]
    
    profile_path = out_dir / 'doc_profile.csv'
    df.to_csv(profile_path, index=False)
    print(f"  Wrote {profile_path}")
    
    # 2. doc_profile_schema.json
    schema_path = out_dir / 'doc_profile_schema.json'
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {schema_path}")
    
    # 3. doc_profile_sanity.json
    sanity_path = out_dir / 'doc_profile_sanity.json'
    with open(sanity_path, 'w', encoding='utf-8') as f:
        json.dump(sanity_report, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {sanity_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total documents: {sanity_report['total_docs']}")
    print(f"Unique tickers: {sanity_report['unique_tickers']}")
    print(f"Year range: {sanity_report['year_range'][0]}-{sanity_report['year_range'][1]}")
    print(f"Docs with no sentences: {sanity_report['docs_with_no_sentences']}")
    print(f"Docs with no tables: {sanity_report['docs_with_no_tables']}")
    print(f"\nAuditability index distribution:")
    audit_dist = sanity_report['distributions']['auditability_index']
    print(f"  Mean: {audit_dist['mean']:.3f}")
    print(f"  Std: {audit_dist['std']:.3f}")
    print(f"  Min: {audit_dist['min']:.3f}")
    print(f"  Median: {audit_dist['p50']:.3f}")
    print(f"  Max: {audit_dist['max']:.3f}")
    print("="*60)
    print("\nDone! Outputs written to:", out_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
