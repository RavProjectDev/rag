"""
Report generation utilities

Handles aggregating results and generating output reports.
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_report(
    documents: List[Any],
    configurations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate comprehensive sanity check report.
    
    Args:
        documents: List of documents checked
        configurations: List of configuration reports
        
    Returns:
        Complete report dictionary
    """
    # Calculate summary statistics
    summary = {
        "total_issues": 0,
        "missing_documents": 0,
        "incomplete_documents": 0,
        "excess_vectors": 0,
        "errors": 0,
        "ok_documents": 0,
    }
    
    for config in configurations:
        for doc_report in config.get("documents", []):
            status = doc_report.get("status", "ERROR")
            
            if status == "OK":
                summary["ok_documents"] += 1
            elif status == "MISSING_ALL":
                summary["missing_documents"] += 1
            elif status == "INCOMPLETE":
                summary["incomplete_documents"] += 1
            elif status == "MISMATCH":
                if doc_report.get("extra_chunks", {}).get("count", 0) > 0:
                    summary["excess_vectors"] += 1
            elif status == "ERROR":
                summary["errors"] += 1
            
            # Count total issues
            missing_count = doc_report.get("missing_chunks", {}).get("count", 0)
            extra_count = doc_report.get("extra_chunks", {}).get("count", 0)
            summary["total_issues"] += missing_count + extra_count
    
    # Build full report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(documents),
        "configurations_checked": len(configurations),
        "summary": summary,
        "configurations": configurations,
    }
    
    return report


def save_report(report: Dict[str, Any], output_file: str):
    """
    Save report to JSON file.
    
    Args:
        report: Report dictionary
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"[REPORT] Saved detailed report to: {output_file}")
    except Exception as e:
        logger.error(f"[REPORT] Failed to save report: {e}")


def print_summary(report: Dict[str, Any]):
    """
    Print human-readable summary to console.
    
    Args:
        report: Report dictionary
    """
    summary = report["summary"]
    
    logger.info(
        f"\n{'='*80}\n"
        f"[SANITY CHECK] FINAL SUMMARY\n"
        f"{'='*80}\n"
        f"Total Documents: {report['total_documents']}\n"
        f"Configurations Checked: {report['configurations_checked']}\n"
        f"Total Issues: {summary['total_issues']}\n"
        f"  - OK Documents: {summary['ok_documents']}\n"
        f"  - Missing Documents: {summary['missing_documents']}\n"
        f"  - Incomplete Documents: {summary['incomplete_documents']}\n"
        f"  - Excess Vectors: {summary['excess_vectors']}\n"
        f"  - Errors: {summary['errors']}\n"
        f"{'='*80}"
    )
    
    # Print per-configuration summary
    for config in report["configurations"]:
        logger.info(
            f"\n[CONFIG] {config['embedding_config']} / {config['chunking_strategy']}\n"
            f"  Index: {config['index_name']}\n"
            f"  Namespace: {config['namespace']}\n"
            f"  Pinecone Vectors: {config['pinecone_stats'].get('vector_count', 'N/A')}\n"
            f"  Documents Checked: {len(config['documents'])}\n"
            f"  Summary: {config['summary']}"
        )


def print_missing_chunks_summary(report: Dict[str, Any], limit: int = 10):
    """
    Print summary of missing chunks for quick review.
    
    Args:
        report: Report dictionary
        limit: Maximum number of missing chunks to display per configuration
    """
    logger.info(f"\n{'='*80}\n[MISSING CHUNKS SUMMARY]\n{'='*80}")
    
    for config in report["configurations"]:
        has_missing = False
        
        for doc_report in config["documents"]:
            missing = doc_report.get("missing_chunks", {})
            if missing.get("count", 0) > 0:
                if not has_missing:
                    logger.info(
                        f"\n[{config['embedding_config']} / {config['chunking_strategy']}]"
                    )
                    has_missing = True
                
                logger.info(
                    f"  {doc_report['document_title']}: "
                    f"{missing['count']} missing chunk(s)"
                )
                
                # Show first few missing chunks
                for detail in missing.get("details", [])[:limit]:
                    logger.info(
                        f"    - {detail['chunk_id']}: "
                        f"{detail['text_preview'][:80]}..."
                    )
        
        if not has_missing:
            logger.info(
                f"\n[{config['embedding_config']} / {config['chunking_strategy']}] "
                f"✓ All chunks present"
            )

