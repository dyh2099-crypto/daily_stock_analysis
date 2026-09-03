#!/usr/bin/env python3
"""Loader for the reproducible A-share limit-up backtest.

The implementation is stored as XZ-compressed base64 text parts to keep GitHub
connector writes deterministic. The decoded source is written next to the
artifacts and executed with the original command-line arguments.
"""
from __future__ import annotations

import base64
import hashlib
import lzma
import os
from pathlib import Path
import sys

EXPECTED_SHA256 = "012c26c05bf14aaacad236db04eb7005979af7ed435c8e9c9bba2178fa7a88be"
HERE = Path(__file__).resolve().parent
PART_DIR = HERE / "limitup_backtest_payload"
parts = sorted(PART_DIR.glob("part*.b64"))
if len(parts) != 6:
    raise SystemExit(f"Expected 6 payload parts, found {len(parts)} under {PART_DIR}")
encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
source = lzma.decompress(base64.b64decode(encoded))
actual = hashlib.sha256(source).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f"Decoded source SHA256 mismatch: {actual}")
run_path = HERE / "_decoded_limitup_quotes_daily_backtest.py"
run_path.write_bytes(source)
os.execv(sys.executable, [sys.executable, str(run_path), *sys.argv[1:]])
