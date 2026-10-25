#!/usr/bin/env python3
"""CPU-only per-batch histogram logging, after the existing device-to-host copy.

Diagnostic overlay only. Does not alter acceptance, sampling, or GPU execution.
All TP ranks see identical counts; log only the statistics rank to avoid duplicates.
"""
import ast
import sys
from pathlib import Path


def instrument(path):
    source = path.read_text()
    assert "EAGLE_POSITION" not in source, "diagnostic already applied"
    anchor = '        predict_tokens = []\n        # In adaptive spec-v2'
    assert source.count(anchor) == 1, 'pinned spec-v2 CPU settlement anchor changed'
    addition = '''        # Diagnostic: unconditional accepted prefix at positions 1..3.
        if self.metrics_reporter.is_stats_logging_rank:
            counts = result.num_correct_drafts_per_req_cpu
            histogram = [sum(value == n for value in counts) for n in range(4)]
            logger.info("EAGLE_POSITION histogram=%s", histogram)

'''
    source = source.replace(anchor, addition + anchor)
    ast.parse(source)
    path.write_text(source)


if __name__ == '__main__':
    instrument(Path(sys.argv[1]))
