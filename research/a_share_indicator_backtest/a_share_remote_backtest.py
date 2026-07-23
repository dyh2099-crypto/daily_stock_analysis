#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-extracting launcher for the readable A-share backtest source.

Decoded-source SHA-256: b513bb195d69fdbbaa74a38d77e8640a4b931e7f1aa65fded8446c04d078d8b8
"""
from __future__ import annotations
import base64
import lzma
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PAYLOAD = "".join(p.read_text(encoding="ascii").strip() for p in sorted(_HERE.glob("a_share_payload_*.txt")))
_SOURCE = lzma.decompress(base64.b64decode(_PAYLOAD.encode("ascii")))
_SOURCE_PATH = _HERE / "a_share_remote_backtest_source.py"
_SOURCE_PATH.write_bytes(_SOURCE)
exec(compile(_SOURCE, str(_SOURCE_PATH), "exec"), globals(), globals())
