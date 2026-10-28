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
    anchor = '        if req.finished():\n            # isinstance narrowing: create_worker may also return plain'
    assert source.count(anchor) == 1, 'pinned completed-request anchor changed'
    addition = '''        if req.finished() and req.spec_verify_ct and self.metrics_reporter.is_stats_logging_rank:
            logger.info("EAGLE_REQUEST rid=%s verify_steps=%d accepted_draft_tokens=%d", req.rid, req.spec_verify_ct, req.spec_num_correct_drafts)

'''
    source = source.replace(anchor, addition + anchor)
    ast.parse(source)
    path.write_text(source)


if __name__ == '__main__':
    instrument(Path(sys.argv[1]))
