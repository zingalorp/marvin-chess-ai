#!/usr/bin/env python3
"""
Export a Marvin (Chessformer) model to Leela Chess Zero's .pb.gz format.

This script:
1. Loads the Marvin .pt checkpoint
2. Wraps it in an adapter that converts Leela's 112-plane input → Marvin's
   token format, and remaps Marvin's 4098-policy output → Leela's 1858-move
   encoding (merging the separate promotion head).
3. Exports the wrapped model to ONNX
4. Wraps the ONNX bytes in lc0's protobuf container (.pb.gz)

Usage:
    python export_to_leela.py \
        --checkpoint inference/marvin_large.pt \
        --output marvin_for_leela.pb.gz \
        [--config auto] [--device cpu]
"""

from __future__ import annotations

import argparse
import gzip
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Leela 1858-move table (from lc0/src/neural/encoder.cc kMoveStrs[])
# =============================================================================
# fmt: off
LC0_MOVE_STRS: List[str] = [
    "a1b1","a1c1","a1d1","a1e1","a1f1","a1g1","a1h1","a1a2",
    "a1b2","a1c2","a1a3","a1b3","a1c3","a1a4","a1d4","a1a5",
    "a1e5","a1a6","a1f6","a1a7","a1g7","a1a8","a1h8","b1a1",
    "b1c1","b1d1","b1e1","b1f1","b1g1","b1h1","b1a2","b1b2",
    "b1c2","b1d2","b1a3","b1b3","b1c3","b1d3","b1b4","b1e4",
    "b1b5","b1f5","b1b6","b1g6","b1b7","b1h7","b1b8","c1a1",
    "c1b1","c1d1","c1e1","c1f1","c1g1","c1h1","c1a2","c1b2",
    "c1c2","c1d2","c1e2","c1a3","c1b3","c1c3","c1d3","c1e3",
    "c1c4","c1f4","c1c5","c1g5","c1c6","c1h6","c1c7","c1c8",
    "d1a1","d1b1","d1c1","d1e1","d1f1","d1g1","d1h1","d1b2",
    "d1c2","d1d2","d1e2","d1f2","d1b3","d1c3","d1d3","d1e3",
    "d1f3","d1a4","d1d4","d1g4","d1d5","d1h5","d1d6","d1d7",
    "d1d8","e1a1","e1b1","e1c1","e1d1","e1f1","e1g1","e1h1",
    "e1c2","e1d2","e1e2","e1f2","e1g2","e1c3","e1d3","e1e3",
    "e1f3","e1g3","e1b4","e1e4","e1h4","e1a5","e1e5","e1e6",
    "e1e7","e1e8","f1a1","f1b1","f1c1","f1d1","f1e1","f1g1",
    "f1h1","f1d2","f1e2","f1f2","f1g2","f1h2","f1d3","f1e3",
    "f1f3","f1g3","f1h3","f1c4","f1f4","f1b5","f1f5","f1a6",
    "f1f6","f1f7","f1f8","g1a1","g1b1","g1c1","g1d1","g1e1",
    "g1f1","g1h1","g1e2","g1f2","g1g2","g1h2","g1e3","g1f3",
    "g1g3","g1h3","g1d4","g1g4","g1c5","g1g5","g1b6","g1g6",
    "g1a7","g1g7","g1g8","h1a1","h1b1","h1c1","h1d1","h1e1",
    "h1f1","h1g1","h1f2","h1g2","h1h2","h1f3","h1g3","h1h3",
    "h1e4","h1h4","h1d5","h1h5","h1c6","h1h6","h1b7","h1h7",
    "h1a8","h1h8","a2a1","a2b1","a2c1","a2b2","a2c2","a2d2",
    "a2e2","a2f2","a2g2","a2h2","a2a3","a2b3","a2c3","a2a4",
    "a2b4","a2c4","a2a5","a2d5","a2a6","a2e6","a2a7","a2f7",
    "a2a8","a2g8","b2a1","b2b1","b2c1","b2d1","b2a2","b2c2",
    "b2d2","b2e2","b2f2","b2g2","b2h2","b2a3","b2b3","b2c3",
    "b2d3","b2a4","b2b4","b2c4","b2d4","b2b5","b2e5","b2b6",
    "b2f6","b2b7","b2g7","b2b8","b2h8","c2a1","c2b1","c2c1",
    "c2d1","c2e1","c2a2","c2b2","c2d2","c2e2","c2f2","c2g2",
    "c2h2","c2a3","c2b3","c2c3","c2d3","c2e3","c2a4","c2b4",
    "c2c4","c2d4","c2e4","c2c5","c2f5","c2c6","c2g6","c2c7",
    "c2h7","c2c8","d2b1","d2c1","d2d1","d2e1","d2f1","d2a2",
    "d2b2","d2c2","d2e2","d2f2","d2g2","d2h2","d2b3","d2c3",
    "d2d3","d2e3","d2f3","d2b4","d2c4","d2d4","d2e4","d2f4",
    "d2a5","d2d5","d2g5","d2d6","d2h6","d2d7","d2d8","e2c1",
    "e2d1","e2e1","e2f1","e2g1","e2a2","e2b2","e2c2","e2d2",
    "e2f2","e2g2","e2h2","e2c3","e2d3","e2e3","e2f3","e2g3",
    "e2c4","e2d4","e2e4","e2f4","e2g4","e2b5","e2e5","e2h5",
    "e2a6","e2e6","e2e7","e2e8","f2d1","f2e1","f2f1","f2g1",
    "f2h1","f2a2","f2b2","f2c2","f2d2","f2e2","f2g2","f2h2",
    "f2d3","f2e3","f2f3","f2g3","f2h3","f2d4","f2e4","f2f4",
    "f2g4","f2h4","f2c5","f2f5","f2b6","f2f6","f2a7","f2f7",
    "f2f8","g2e1","g2f1","g2g1","g2h1","g2a2","g2b2","g2c2",
    "g2d2","g2e2","g2f2","g2h2","g2e3","g2f3","g2g3","g2h3",
    "g2e4","g2f4","g2g4","g2h4","g2d5","g2g5","g2c6","g2g6",
    "g2b7","g2g7","g2a8","g2g8","h2f1","h2g1","h2h1","h2a2",
    "h2b2","h2c2","h2d2","h2e2","h2f2","h2g2","h2f3","h2g3",
    "h2h3","h2f4","h2g4","h2h4","h2e5","h2h5","h2d6","h2h6",
    "h2c7","h2h7","h2b8","h2h8","a3a1","a3b1","a3c1","a3a2",
    "a3b2","a3c2","a3b3","a3c3","a3d3","a3e3","a3f3","a3g3",
    "a3h3","a3a4","a3b4","a3c4","a3a5","a3b5","a3c5","a3a6",
    "a3d6","a3a7","a3e7","a3a8","a3f8","b3a1","b3b1","b3c1",
    "b3d1","b3a2","b3b2","b3c2","b3d2","b3a3","b3c3","b3d3",
    "b3e3","b3f3","b3g3","b3h3","b3a4","b3b4","b3c4","b3d4",
    "b3a5","b3b5","b3c5","b3d5","b3b6","b3e6","b3b7","b3f7",
    "b3b8","b3g8","c3a1","c3b1","c3c1","c3d1","c3e1","c3a2",
    "c3b2","c3c2","c3d2","c3e2","c3a3","c3b3","c3d3","c3e3",
    "c3f3","c3g3","c3h3","c3a4","c3b4","c3c4","c3d4","c3e4",
    "c3a5","c3b5","c3c5","c3d5","c3e5","c3c6","c3f6","c3c7",
    "c3g7","c3c8","c3h8","d3b1","d3c1","d3d1","d3e1","d3f1",
    "d3b2","d3c2","d3d2","d3e2","d3f2","d3a3","d3b3","d3c3",
    "d3e3","d3f3","d3g3","d3h3","d3b4","d3c4","d3d4","d3e4",
    "d3f4","d3b5","d3c5","d3d5","d3e5","d3f5","d3a6","d3d6",
    "d3g6","d3d7","d3h7","d3d8","e3c1","e3d1","e3e1","e3f1",
    "e3g1","e3c2","e3d2","e3e2","e3f2","e3g2","e3a3","e3b3",
    "e3c3","e3d3","e3f3","e3g3","e3h3","e3c4","e3d4","e3e4",
    "e3f4","e3g4","e3c5","e3d5","e3e5","e3f5","e3g5","e3b6",
    "e3e6","e3h6","e3a7","e3e7","e3e8","f3d1","f3e1","f3f1",
    "f3g1","f3h1","f3d2","f3e2","f3f2","f3g2","f3h2","f3a3",
    "f3b3","f3c3","f3d3","f3e3","f3g3","f3h3","f3d4","f3e4",
    "f3f4","f3g4","f3h4","f3d5","f3e5","f3f5","f3g5","f3h5",
    "f3c6","f3f6","f3b7","f3f7","f3a8","f3f8","g3e1","g3f1",
    "g3g1","g3h1","g3e2","g3f2","g3g2","g3h2","g3a3","g3b3",
    "g3c3","g3d3","g3e3","g3f3","g3h3","g3e4","g3f4","g3g4",
    "g3h4","g3e5","g3f5","g3g5","g3h5","g3d6","g3g6","g3c7",
    "g3g7","g3b8","g3g8","h3f1","h3g1","h3h1","h3f2","h3g2",
    "h3h2","h3a3","h3b3","h3c3","h3d3","h3e3","h3f3","h3g3",
    "h3f4","h3g4","h3h4","h3f5","h3g5","h3h5","h3e6","h3h6",
    "h3d7","h3h7","h3c8","h3h8","a4a1","a4d1","a4a2","a4b2",
    "a4c2","a4a3","a4b3","a4c3","a4b4","a4c4","a4d4","a4e4",
    "a4f4","a4g4","a4h4","a4a5","a4b5","a4c5","a4a6","a4b6",
    "a4c6","a4a7","a4d7","a4a8","a4e8","b4b1","b4e1","b4a2",
    "b4b2","b4c2","b4d2","b4a3","b4b3","b4c3","b4d3","b4a4",
    "b4c4","b4d4","b4e4","b4f4","b4g4","b4h4","b4a5","b4b5",
    "b4c5","b4d5","b4a6","b4b6","b4c6","b4d6","b4b7","b4e7",
    "b4b8","b4f8","c4c1","c4f1","c4a2","c4b2","c4c2","c4d2",
    "c4e2","c4a3","c4b3","c4c3","c4d3","c4e3","c4a4","c4b4",
    "c4d4","c4e4","c4f4","c4g4","c4h4","c4a5","c4b5","c4c5",
    "c4d5","c4e5","c4a6","c4b6","c4c6","c4d6","c4e6","c4c7",
    "c4f7","c4c8","c4g8","d4a1","d4d1","d4g1","d4b2","d4c2",
    "d4d2","d4e2","d4f2","d4b3","d4c3","d4d3","d4e3","d4f3",
    "d4a4","d4b4","d4c4","d4e4","d4f4","d4g4","d4h4","d4b5",
    "d4c5","d4d5","d4e5","d4f5","d4b6","d4c6","d4d6","d4e6",
    "d4f6","d4a7","d4d7","d4g7","d4d8","d4h8","e4b1","e4e1",
    "e4h1","e4c2","e4d2","e4e2","e4f2","e4g2","e4c3","e4d3",
    "e4e3","e4f3","e4g3","e4a4","e4b4","e4c4","e4d4","e4f4",
    "e4g4","e4h4","e4c5","e4d5","e4e5","e4f5","e4g5","e4c6",
    "e4d6","e4e6","e4f6","e4g6","e4b7","e4e7","e4h7","e4a8",
    "e4e8","f4c1","f4f1","f4d2","f4e2","f4f2","f4g2","f4h2",
    "f4d3","f4e3","f4f3","f4g3","f4h3","f4a4","f4b4","f4c4",
    "f4d4","f4e4","f4g4","f4h4","f4d5","f4e5","f4f5","f4g5",
    "f4h5","f4d6","f4e6","f4f6","f4g6","f4h6","f4c7","f4f7",
    "f4b8","f4f8","g4d1","g4g1","g4e2","g4f2","g4g2","g4h2",
    "g4e3","g4f3","g4g3","g4h3","g4a4","g4b4","g4c4","g4d4",
    "g4e4","g4f4","g4h4","g4e5","g4f5","g4g5","g4h5","g4e6",
    "g4f6","g4g6","g4h6","g4d7","g4g7","g4c8","g4g8","h4e1",
    "h4h1","h4f2","h4g2","h4h2","h4f3","h4g3","h4h3","h4a4",
    "h4b4","h4c4","h4d4","h4e4","h4f4","h4g4","h4f5","h4g5",
    "h4h5","h4f6","h4g6","h4h6","h4e7","h4h7","h4d8","h4h8",
    "a5a1","a5e1","a5a2","a5d2","a5a3","a5b3","a5c3","a5a4",
    "a5b4","a5c4","a5b5","a5c5","a5d5","a5e5","a5f5","a5g5",
    "a5h5","a5a6","a5b6","a5c6","a5a7","a5b7","a5c7","a5a8",
    "a5d8","b5b1","b5f1","b5b2","b5e2","b5a3","b5b3","b5c3",
    "b5d3","b5a4","b5b4","b5c4","b5d4","b5a5","b5c5","b5d5",
    "b5e5","b5f5","b5g5","b5h5","b5a6","b5b6","b5c6","b5d6",
    "b5a7","b5b7","b5c7","b5d7","b5b8","b5e8","c5c1","c5g1",
    "c5c2","c5f2","c5a3","c5b3","c5c3","c5d3","c5e3","c5a4",
    "c5b4","c5c4","c5d4","c5e4","c5a5","c5b5","c5d5","c5e5",
    "c5f5","c5g5","c5h5","c5a6","c5b6","c5c6","c5d6","c5e6",
    "c5a7","c5b7","c5c7","c5d7","c5e7","c5c8","c5f8","d5d1",
    "d5h1","d5a2","d5d2","d5g2","d5b3","d5c3","d5d3","d5e3",
    "d5f3","d5b4","d5c4","d5d4","d5e4","d5f4","d5a5","d5b5",
    "d5c5","d5e5","d5f5","d5g5","d5h5","d5b6","d5c6","d5d6",
    "d5e6","d5f6","d5b7","d5c7","d5d7","d5e7","d5f7","d5a8",
    "d5d8","d5g8","e5a1","e5e1","e5b2","e5e2","e5h2","e5c3",
    "e5d3","e5e3","e5f3","e5g3","e5c4","e5d4","e5e4","e5f4",
    "e5g4","e5a5","e5b5","e5c5","e5d5","e5f5","e5g5","e5h5",
    "e5c6","e5d6","e5e6","e5f6","e5g6","e5c7","e5d7","e5e7",
    "e5f7","e5g7","e5b8","e5e8","e5h8","f5b1","f5f1","f5c2",
    "f5f2","f5d3","f5e3","f5f3","f5g3","f5h3","f5d4","f5e4",
    "f5f4","f5g4","f5h4","f5a5","f5b5","f5c5","f5d5","f5e5",
    "f5g5","f5h5","f5d6","f5e6","f5f6","f5g6","f5h6","f5d7",
    "f5e7","f5f7","f5g7","f5h7","f5c8","f5f8","g5c1","g5g1",
    "g5d2","g5g2","g5e3","g5f3","g5g3","g5h3","g5e4","g5f4",
    "g5g4","g5h4","g5a5","g5b5","g5c5","g5d5","g5e5","g5f5",
    "g5h5","g5e6","g5f6","g5g6","g5h6","g5e7","g5f7","g5g7",
    "g5h7","g5d8","g5g8","h5d1","h5h1","h5e2","h5h2","h5f3",
    "h5g3","h5h3","h5f4","h5g4","h5h4","h5a5","h5b5","h5c5",
    "h5d5","h5e5","h5f5","h5g5","h5f6","h5g6","h5h6","h5f7",
    "h5g7","h5h7","h5e8","h5h8","a6a1","a6f1","a6a2","a6e2",
    "a6a3","a6d3","a6a4","a6b4","a6c4","a6a5","a6b5","a6c5",
    "a6b6","a6c6","a6d6","a6e6","a6f6","a6g6","a6h6","a6a7",
    "a6b7","a6c7","a6a8","a6b8","a6c8","b6b1","b6g1","b6b2",
    "b6f2","b6b3","b6e3","b6a4","b6b4","b6c4","b6d4","b6a5",
    "b6b5","b6c5","b6d5","b6a6","b6c6","b6d6","b6e6","b6f6",
    "b6g6","b6h6","b6a7","b6b7","b6c7","b6d7","b6a8","b6b8",
    "b6c8","b6d8","c6c1","c6h1","c6c2","c6g2","c6c3","c6f3",
    "c6a4","c6b4","c6c4","c6d4","c6e4","c6a5","c6b5","c6c5",
    "c6d5","c6e5","c6a6","c6b6","c6d6","c6e6","c6f6","c6g6",
    "c6h6","c6a7","c6b7","c6c7","c6d7","c6e7","c6a8","c6b8",
    "c6c8","c6d8","c6e8","d6d1","d6d2","d6h2","d6a3","d6d3",
    "d6g3","d6b4","d6c4","d6d4","d6e4","d6f4","d6b5","d6c5",
    "d6d5","d6e5","d6f5","d6a6","d6b6","d6c6","d6e6","d6f6",
    "d6g6","d6h6","d6b7","d6c7","d6d7","d6e7","d6f7","d6b8",
    "d6c8","d6d8","d6e8","d6f8","e6e1","e6a2","e6e2","e6b3",
    "e6e3","e6h3","e6c4","e6d4","e6e4","e6f4","e6g4","e6c5",
    "e6d5","e6e5","e6f5","e6g5","e6a6","e6b6","e6c6","e6d6",
    "e6f6","e6g6","e6h6","e6c7","e6d7","e6e7","e6f7","e6g7",
    "e6c8","e6d8","e6e8","e6f8","e6g8","f6a1","f6f1","f6b2",
    "f6f2","f6c3","f6f3","f6d4","f6e4","f6f4","f6g4","f6h4",
    "f6d5","f6e5","f6f5","f6g5","f6h5","f6a6","f6b6","f6c6",
    "f6d6","f6e6","f6g6","f6h6","f6d7","f6e7","f6f7","f6g7",
    "f6h7","f6d8","f6e8","f6f8","f6g8","f6h8","g6b1","g6g1",
    "g6c2","g6g2","g6d3","g6g3","g6e4","g6f4","g6g4","g6h4",
    "g6e5","g6f5","g6g5","g6h5","g6a6","g6b6","g6c6","g6d6",
    "g6e6","g6f6","g6h6","g6e7","g6f7","g6g7","g6h7","g6e8",
    "g6f8","g6g8","g6h8","h6c1","h6h1","h6d2","h6h2","h6e3",
    "h6h3","h6f4","h6g4","h6h4","h6f5","h6g5","h6h5","h6a6",
    "h6b6","h6c6","h6d6","h6e6","h6f6","h6g6","h6f7","h6g7",
    "h6h7","h6f8","h6g8","h6h8","a7a1","a7g1","a7a2","a7f2",
    "a7a3","a7e3","a7a4","a7d4","a7a5","a7b5","a7c5","a7a6",
    "a7b6","a7c6","a7b7","a7c7","a7d7","a7e7","a7f7","a7g7",
    "a7h7","a7a8","a7b8","a7c8","b7b1","b7h1","b7b2","b7g2",
    "b7b3","b7f3","b7b4","b7e4","b7a5","b7b5","b7c5","b7d5",
    "b7a6","b7b6","b7c6","b7d6","b7a7","b7c7","b7d7","b7e7",
    "b7f7","b7g7","b7h7","b7a8","b7b8","b7c8","b7d8","c7c1",
    "c7c2","c7h2","c7c3","c7g3","c7c4","c7f4","c7a5","c7b5",
    "c7c5","c7d5","c7e5","c7a6","c7b6","c7c6","c7d6","c7e6",
    "c7a7","c7b7","c7d7","c7e7","c7f7","c7g7","c7h7","c7a8",
    "c7b8","c7c8","c7d8","c7e8","d7d1","d7d2","d7d3","d7h3",
    "d7a4","d7d4","d7g4","d7b5","d7c5","d7d5","d7e5","d7f5",
    "d7b6","d7c6","d7d6","d7e6","d7f6","d7a7","d7b7","d7c7",
    "d7e7","d7f7","d7g7","d7h7","d7b8","d7c8","d7d8","d7e8",
    "d7f8","e7e1","e7e2","e7a3","e7e3","e7b4","e7e4","e7h4",
    "e7c5","e7d5","e7e5","e7f5","e7g5","e7c6","e7d6","e7e6",
    "e7f6","e7g6","e7a7","e7b7","e7c7","e7d7","e7f7","e7g7",
    "e7h7","e7c8","e7d8","e7e8","e7f8","e7g8","f7f1","f7a2",
    "f7f2","f7b3","f7f3","f7c4","f7f4","f7d5","f7e5","f7f5",
    "f7g5","f7h5","f7d6","f7e6","f7f6","f7g6","f7h6","f7a7",
    "f7b7","f7c7","f7d7","f7e7","f7g7","f7h7","f7d8","f7e8",
    "f7f8","f7g8","f7h8","g7a1","g7g1","g7b2","g7g2","g7c3",
    "g7g3","g7d4","g7g4","g7e5","g7f5","g7g5","g7h5","g7e6",
    "g7f6","g7g6","g7h6","g7a7","g7b7","g7c7","g7d7","g7e7",
    "g7f7","g7h7","g7e8","g7f8","g7g8","g7h8","h7b1","h7h1",
    "h7c2","h7h2","h7d3","h7h3","h7e4","h7h4","h7f5","h7g5",
    "h7h5","h7f6","h7g6","h7h6","h7a7","h7b7","h7c7","h7d7",
    "h7e7","h7f7","h7g7","h7f8","h7g8","h7h8","a8a1","a8h1",
    "a8a2","a8g2","a8a3","a8f3","a8a4","a8e4","a8a5","a8d5",
    "a8a6","a8b6","a8c6","a8a7","a8b7","a8c7","a8b8","a8c8",
    "a8d8","a8e8","a8f8","a8g8","a8h8","b8b1","b8b2","b8h2",
    "b8b3","b8g3","b8b4","b8f4","b8b5","b8e5","b8a6","b8b6",
    "b8c6","b8d6","b8a7","b8b7","b8c7","b8d7","b8a8","b8c8",
    "b8d8","b8e8","b8f8","b8g8","b8h8","c8c1","c8c2","c8c3",
    "c8h3","c8c4","c8g4","c8c5","c8f5","c8a6","c8b6","c8c6",
    "c8d6","c8e6","c8a7","c8b7","c8c7","c8d7","c8e7","c8a8",
    "c8b8","c8d8","c8e8","c8f8","c8g8","c8h8","d8d1","d8d2",
    "d8d3","d8d4","d8h4","d8a5","d8d5","d8g5","d8b6","d8c6",
    "d8d6","d8e6","d8f6","d8b7","d8c7","d8d7","d8e7","d8f7",
    "d8a8","d8b8","d8c8","d8e8","d8f8","d8g8","d8h8","e8e1",
    "e8e2","e8e3","e8a4","e8e4","e8b5","e8e5","e8h5","e8c6",
    "e8d6","e8e6","e8f6","e8g6","e8c7","e8d7","e8e7","e8f7",
    "e8g7","e8a8","e8b8","e8c8","e8d8","e8f8","e8g8","e8h8",
    "f8f1","f8f2","f8a3","f8f3","f8b4","f8f4","f8c5","f8f5",
    "f8d6","f8e6","f8f6","f8g6","f8h6","f8d7","f8e7","f8f7",
    "f8g7","f8h7","f8a8","f8b8","f8c8","f8d8","f8e8","f8g8",
    "f8h8","g8g1","g8a2","g8g2","g8b3","g8g3","g8c4","g8g4",
    "g8d5","g8g5","g8e6","g8f6","g8g6","g8h6","g8e7","g8f7",
    "g8g7","g8h7","g8a8","g8b8","g8c8","g8d8","g8e8","g8f8",
    "g8h8","h8a1","h8h1","h8b2","h8h2","h8c3","h8h3","h8d4",
    "h8h4","h8e5","h8h5","h8f6","h8g6","h8h6","h8f7","h8g7",
    "h8h7","h8a8","h8b8","h8c8","h8d8","h8e8","h8f8","h8g8",
    # Promotions: from 7th rank to 8th rank with piece suffix
    "a7a8q","a7a8r","a7a8b","a7b8q","a7b8r","a7b8b",
    "b7a8q","b7a8r","b7a8b","b7b8q","b7b8r","b7b8b","b7c8q","b7c8r","b7c8b",
    "c7b8q","c7b8r","c7b8b","c7c8q","c7c8r","c7c8b","c7d8q","c7d8r","c7d8b",
    "d7c8q","d7c8r","d7c8b","d7d8q","d7d8r","d7d8b","d7e8q","d7e8r","d7e8b",
    "e7d8q","e7d8r","e7d8b","e7e8q","e7e8r","e7e8b","e7f8q","e7f8r","e7f8b",
    "f7e8q","f7e8r","f7e8b","f7f8q","f7f8r","f7f8b","f7g8q","f7g8r","f7g8b",
    "g7f8q","g7f8r","g7f8b","g7g8q","g7g8r","g7g8b","g7h8q","g7h8r","g7h8b",
    "h7g8q","h7g8r","h7g8b","h7h8q","h7h8r","h7h8b",
]
# fmt: on

assert len(LC0_MOVE_STRS) == 1858, f"Expected 1858 moves, got {len(LC0_MOVE_STRS)}"


# =============================================================================
# Build mapping tables between Leela 1858 encoding and Marvin 4098 encoding
# =============================================================================

def _sq_name_to_idx(name: str) -> int:
    """Convert square name like 'a1' to 0-63 index (a1=0, b1=1, ..., h8=63)."""
    file = ord(name[0]) - ord('a')  # 0-7
    rank = int(name[1]) - 1          # 0-7
    return rank * 8 + file


def _parse_lc0_move(move_str: str) -> Tuple[int, int, Optional[str]]:
    """Parse lc0 move string -> (from_sq, to_sq, promo_piece_or_None)."""
    from_sq = _sq_name_to_idx(move_str[:2])
    to_sq = _sq_name_to_idx(move_str[2:4])
    promo = move_str[4] if len(move_str) == 5 else None
    return from_sq, to_sq, promo


def build_lc0_to_marvin_mapping() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build index mapping from lc0's 1858 policy indices to Marvin's encoding.

    Returns:
        marvin_base_idx: (1858,) int64 - Marvin's from*64+to index for each lc0 move
        is_promotion: (1858,) bool - whether this lc0 move is a promotion
        promo_piece_idx: (1858,) int64 - promotion piece index for Marvin's promo head
            0=Queen, 1=Rook, 2=Bishop, 3=Knight (matches Marvin's promo_bias_proj output)
    """
    marvin_base_idx = torch.zeros(1858, dtype=torch.long)
    is_promotion = torch.zeros(1858, dtype=torch.bool)
    promo_piece_idx = torch.zeros(1858, dtype=torch.long)

    # Marvin's promotion head outputs [Q, R, B, N] for each file on rank 8
    promo_map = {'q': 0, 'r': 1, 'b': 2, 'n': 3}

    for lc0_idx, move_str in enumerate(LC0_MOVE_STRS):
        from_sq, to_sq, promo = _parse_lc0_move(move_str)
        marvin_base_idx[lc0_idx] = from_sq * 64 + to_sq

        if promo is not None:
            is_promotion[lc0_idx] = True
            promo_piece_idx[lc0_idx] = promo_map[promo]

    return marvin_base_idx, is_promotion, promo_piece_idx


# =============================================================================
# Leela piece plane ordering -> Marvin piece code mapping
# =============================================================================
# Leela's 13 planes per history step (from encoder.cc):
#   0: our pawns      -> Marvin piece_code depends on perspective
#   1: our knights
#   2: our bishops
#   3: our rooks
#   4: our queens
#   5: our king
#   6: their pawns
#   7: their knights
#   8: their bishops
#   9: their rooks
#  10: their queens
#  11: their king
#  12: repetition (all 1s if repeated >= 1)
#
# Marvin piece codes (always canonical = white-to-move):
#   0: empty
#   1: White Pawn, 2: White Knight, 3: White Bishop, 4: White Rook, 5: White Queen, 6: White King
#   7: Black Pawn, 8: Black Knight, 9: Black Bishop, 10: Black Rook, 11: Black Queen, 12: Black King
#
# Since both use canonical (side-to-move = white), Leela's "our" = white, "their" = black.
# Leela plane 0 (our pawns) -> white pawn -> Marvin code 1
# Leela plane 6 (their pawns) -> black pawn -> Marvin code 7
# Leela plane 12 (repetition) -> handled separately


# =============================================================================
# The Adapter Model
# =============================================================================

class MarvinLeelaAdapter(nn.Module):
    """
    Wraps a Marvin Chessformer model to accept Leela's 112-plane input
    and output Leela's 1858-move policy + WDL value.

    Input: (B, 112, 8, 8) float32 — Leela's INPUT_CLASSICAL_112_PLANE format
    Outputs:
        /output/policy: (B, 1858) float32 — policy logits in lc0 move encoding
        /output/wdl:    (B, 3) float32 — [Win, Draw, Loss] probabilities (softmax)
    """

    # Hardcoded defaults for inputs Leela doesn't provide
    DEFAULT_ELO = 2300
    DEFAULT_CLOCK_S = 300.0
    DEFAULT_INC_S = 0.0

    def __init__(self, marvin_model: nn.Module):
        super().__init__()
        self.marvin = marvin_model
        self.marvin.eval()

        # Freeze all Marvin parameters (we're just wrapping, not training)
        for p in self.marvin.parameters():
            p.requires_grad = False

        # Register mapping buffers
        marvin_base_idx, is_promotion, promo_piece_idx = build_lc0_to_marvin_mapping()
        self.register_buffer("marvin_base_idx", marvin_base_idx)
        self.register_buffer("is_promotion", is_promotion)
        self.register_buffer("promo_piece_idx", promo_piece_idx)

        # Precompute which destination file (0-7) each promotion move targets
        # For promo head: index into (B, 8, 4) where 8 = files a-h on rank 8
        promo_dest_file = torch.zeros(1858, dtype=torch.long)
        for lc0_idx, move_str in enumerate(LC0_MOVE_STRS):
            if len(move_str) == 5:  # promotion
                to_sq = _sq_name_to_idx(move_str[2:4])
                to_file = to_sq % 8
                promo_dest_file[lc0_idx] = to_file
        self.register_buffer("promo_dest_file", promo_dest_file)

    def _decode_leela_planes(self, planes: torch.Tensor) -> dict:
        """
        Convert Leela's (B, 112, 8, 8) input to Marvin's batch dict.

        Leela INPUT_CLASSICAL_112_PLANE layout:
          Planes 0-12:   History step 0 (current position) — 13 planes
          Planes 13-25:  History step 1 — 13 planes
          ...
          Planes 91-103: History step 7 — 13 planes
          Plane 104: our queenside castling (all 1s if can castle)
          Plane 105: our kingside castling (all 1s if can castle)
          Plane 106: their queenside castling
          Plane 107: their kingside castling
          Plane 108: side to move (all 1s if black — but in classical, always 0 since already flipped)
          Plane 109: rule50 count (all squares filled with ply count value)
          Plane 110: move count (all zeros in modern lc0)
          Plane 111: all ones
        """
        B = planes.shape[0]
        device = planes.device

        # --- Extract board history as piece codes ---
        # For each history step, reconstruct the 64-square piece code array
        board_history = torch.zeros(B, 8, 64, dtype=torch.long, device=device)

        for hist_idx in range(8):
            base = hist_idx * 13
            # Planes base+0..base+11 are piece bitplanes
            # Our pieces (planes 0-5): P=1, N=2, B=3, R=4, Q=5, K=6
            # Their pieces (planes 6-11): P=7, N=8, B=9, R=10, Q=11, K=12
            piece_planes = planes[:, base:base + 12, :, :]  # (B, 12, 8, 8)
            # Flatten spatial: (B, 12, 64)
            piece_flat = piece_planes.reshape(B, 12, 64)

            # Prepend an "empty" channel (all zeros) so argmax gives 0 = empty
            empty_channel = torch.zeros(B, 1, 64, device=device)
            # Stack: channel 0 = empty, channels 1-12 = piece types
            all_channels = torch.cat([empty_channel, piece_flat], dim=1)  # (B, 13, 64)
            # Take argmax across the 13 channels -> piece code 0-12
            board_history[:, hist_idx, :] = all_channels.argmax(dim=1)  # (B, 64)

            # Note: plane base+12 is repetition — handled separately below

        # --- Repetition flags ---
        rep_flags = torch.zeros(B, 8, dtype=torch.float32, device=device)
        for hist_idx in range(8):
            rep_plane = planes[:, hist_idx * 13 + 12, :, :]  # (B, 8, 8)
            # If all squares are 1, this position has been repeated
            rep_flags[:, hist_idx] = (rep_plane.reshape(B, 64).mean(dim=1) > 0.5).float()

        # --- Castling rights ---
        # Leela classical: planes 104-107 are all-1s flags
        # Marvin order: [WK, WQ, BK, BQ]
        # Leela order: [our_queenside(104), our_kingside(105), their_queenside(106), their_kingside(107)]
        # Since canonical (white to move): our=white, their=black
        castling = torch.zeros(B, 4, dtype=torch.float32, device=device)
        castling[:, 0] = (planes[:, 105, 0, 0] > 0.5).float()  # WK = our kingside
        castling[:, 1] = (planes[:, 104, 0, 0] > 0.5).float()  # WQ = our queenside
        castling[:, 2] = (planes[:, 107, 0, 0] > 0.5).float()  # BK = their kingside
        castling[:, 3] = (planes[:, 106, 0, 0] > 0.5).float()  # BQ = their queenside

        # --- En passant ---
        # In INPUT_CLASSICAL_112_PLANE, plane 108 is side-to-move indicator, NOT en passant.
        # En passant is only in the canonical formats. In classical format,
        # we must infer EP from the board state — but lc0 doesn't encode it as a plane.
        # We'll set ep_mask to all zeros (EP info is lost in classical format).
        # NOTE: This is a known limitation. Use INPUT_112_WITH_CASTLING_PLANE or
        # canonical formats for EP support — but those change castling encoding too.
        ep_mask = torch.zeros(B, 64, dtype=torch.float32, device=device)

        # --- Scalars (hardcoded) ---
        # Marvin scalars: [active_elo, opp_elo, ply, active_clock, opp_clock, active_inc, opp_inc, hmc]
        elo_norm = (self.DEFAULT_ELO - 1900) / 700.0
        clock_norm = math.log1p(self.DEFAULT_CLOCK_S) / 10.0
        inc_norm = self.DEFAULT_INC_S / 30.0

        # Get halfmove clock from plane 109 (rule50 count)
        hmc_raw = planes[:, 109, 0, 0]  # value is the raw ply count
        hmc_norm = hmc_raw / 100.0

        # Get ply from plane 110 (move count) — typically 0 in modern lc0
        # Since we don't have ply info, use a reasonable default
        ply_norm = torch.zeros(B, dtype=torch.float32, device=device)

        scalars = torch.stack([
            torch.full((B,), elo_norm, device=device),       # active_elo
            torch.full((B,), elo_norm, device=device),       # opp_elo
            ply_norm,                                         # ply
            torch.full((B,), clock_norm, device=device),     # active_clock
            torch.full((B,), clock_norm, device=device),     # opp_clock
            torch.full((B,), inc_norm, device=device),       # active_inc
            torch.full((B,), inc_norm, device=device),       # opp_inc
            hmc_norm,                                         # halfmove_clock
        ], dim=1)  # (B, 8)

        # --- Time history (hardcoded zeros — Leela doesn't provide this) ---
        time_history = torch.zeros(B, 8, dtype=torch.float32, device=device)

        # --- Time control category ---
        # 300+0 = Blitz (duration=300 < 600)
        tc_cat = torch.zeros(B, dtype=torch.long, device=device)  # 0 = Blitz

        return {
            'board_history': board_history,
            'time_history': time_history,
            'rep_flags': rep_flags,
            'castling': castling,
            'ep_mask': ep_mask,
            'scalars': scalars,
            'tc_cat': tc_cat,
            'legal_mask': None,  # No masking — lc0 handles legality in search
        }

    def _remap_policy(self, marvin_policy: torch.Tensor,
                      promo_logits: torch.Tensor) -> torch.Tensor:
        """
        Remap Marvin's 4098-policy output to lc0's 1858-move encoding.

        For non-promotion moves: just gather from Marvin's from*64+to logits.
        For promotion moves: add Marvin's base from*64+to logit with the
            promotion-specific logit from the promo head.

        Args:
            marvin_policy: (B, 4098) — Marvin's full policy logits
            promo_logits: (B, 8, 4) — promotion logits per file [Q, R, B, N]

        Returns:
            (B, 1858) — policy logits in lc0 encoding
        """
        B = marvin_policy.shape[0]

        # Gather base move logits from Marvin's 4096 from*64+to space
        # marvin_base_idx maps each lc0 move to its Marvin from*64+to index
        base_logits = marvin_policy[:, :4096]  # (B, 4096) — exclude resign/flag
        lc0_logits = torch.gather(
            base_logits, 1,
            self.marvin_base_idx.unsqueeze(0).expand(B, -1)
        )  # (B, 1858)

        # For promotion moves, add the promotion-specific bias from Marvin's promo head
        # promo_logits: (B, 8, 4) — [file_idx][piece_idx]
        # We need to gather the correct (file, piece) for each promotion lc0 move
        promo_file = self.promo_dest_file.unsqueeze(0).expand(B, -1)  # (B, 1858)
        promo_piece = self.promo_piece_idx.unsqueeze(0).expand(B, -1)  # (B, 1858)

        # Gather file first: (B, 1858) -> index into (B, 8, 4)
        promo_per_file = torch.gather(
            promo_logits, 1,
            promo_file.unsqueeze(-1).expand(-1, -1, 4)
        )  # (B, 1858, 4)
        promo_bias = torch.gather(
            promo_per_file, 2,
            promo_piece.unsqueeze(-1)
        ).squeeze(-1)  # (B, 1858)

        # Only add promotion bias for actual promotion moves
        is_promo = self.is_promotion.unsqueeze(0).expand(B, -1).float()
        lc0_logits = lc0_logits + promo_bias * is_promo

        return lc0_logits

    def forward(self, planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            planes: (B, 112, 8, 8) — Leela's input format

        Returns:
            policy: (B, 1858) — policy logits in lc0 move encoding
            wdl: (B, 3) — [Win, Draw, Loss] probabilities (softmax applied)
        """
        # Step 1: Decode Leela planes into Marvin batch format
        batch = self._decode_leela_planes(planes)

        # Step 2: Run Marvin forward pass (with promotion head)
        # Returns: move_logits, value_out, value_cls_out, value_error_out,
        #          time_cls_out, start_square_logits, promo_logits
        outputs = self.marvin(batch, return_promo=True)
        marvin_policy = outputs[0]      # (B, 4098)
        value_cls_out = outputs[2]      # (B, 3) — [Loss, Draw, Win] logits
        promo_logits = outputs[6]       # (B, 8, 4)

        # Step 3: Remap policy from Marvin's 4098 to lc0's 1858
        lc0_policy = self._remap_policy(marvin_policy, promo_logits)

        # Step 4: Convert WDL
        # Marvin outputs [Loss, Draw, Win] logits
        # lc0 expects [Win, Draw, Loss] after softmax
        wdl_probs = F.softmax(value_cls_out, dim=-1)  # (B, 3) as [L, D, W]
        # Reorder to [W, D, L] for lc0
        wdl_lc0 = torch.stack([
            wdl_probs[:, 2],  # Win
            wdl_probs[:, 1],  # Draw
            wdl_probs[:, 0],  # Loss
        ], dim=1)  # (B, 3)

        return lc0_policy, wdl_lc0


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(adapter: MarvinLeelaAdapter, output_path: str,
                   opset_version: int = 17) -> None:
    """Export the adapted model to ONNX format with all weights embedded."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    adapter.eval()
    device = next(adapter.parameters()).device

    # Create dummy input matching Leela's format
    dummy_input = torch.zeros(1, 112, 8, 8, device=device)
    dummy_input[0, 111, :, :] = 1.0  # all-ones plane
    dummy_input[0, 5, 0, 4] = 1.0    # white king on e1
    dummy_input[0, 11, 7, 4] = 1.0   # black king on e8

    # Use dynamic_shapes for the dynamo exporter
    batch_dim = torch.export.Dim("batch", min=1, max=256)
    dynamic_shapes = {"planes": {0: batch_dim}}

    # Try legacy exporter first (better dynamic axes support),
    # fall back to dynamo exporter if that fails
    export_ok = False
    try:
        # Legacy exporter with dynamic_axes
        torch.onnx.export(
            adapter,
            (dummy_input,),
            output_path,
            opset_version=17,
            input_names=["/input/planes"],
            output_names=["/output/policy", "/output/wdl"],
            dynamic_axes={
                "/input/planes": {0: "batch"},
                "/output/policy": {0: "batch"},
                "/output/wdl": {0: "batch"},
            },
            dynamo=False,
        )
        export_ok = True
        print("  Used legacy ONNX exporter")
    except Exception as e:
        print(f"  Legacy exporter failed ({e}), trying dynamo exporter...")

    if not export_ok:
        # Dynamo exporter (may hardcode batch dim — will be fixed below)
        torch.onnx.export(
            adapter,
            (dummy_input,),
            output_path,
            opset_version=max(opset_version, 18),
            input_names=["/input/planes"],
            output_names=["/output/policy", "/output/wdl"],
            dynamic_shapes=dynamic_shapes,
        )

    # Reload and convert external data to embedded (lc0 needs a single file)
    onnx_model = onnx.load(output_path, load_external_data=True)

    # Downgrade IR version to 9 for compatibility with older ONNX Runtime in lc0
    # IR version 10 was set by opset 18 export, but the actual ops are compatible
    if onnx_model.ir_version > 9:
        onnx_model.ir_version = 9

    # Fix batch dimension to be dynamic (dynamo exporter hardcodes it to 1)
    # Step 1: Fix input/output tensor shapes
    for tensor in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        shape = tensor.type.tensor_type.shape
        if shape and len(shape.dim) > 0:
            shape.dim[0].ClearField("dim_value")
            shape.dim[0].dim_param = "batch"

    # Step 2: Fix internal Reshape ops that have hardcoded batch=1
    # Strategy: Extract batch size from input with Shape+Gather, then for each
    # Reshape with [1, ...], replace the shape constant with a dynamic one built
    # by concatenating [batch_size] with the remaining static dims.
    from onnx import helper, numpy_helper, TensorProto
    import numpy as np

    input_name = onnx_model.graph.input[0].name  # "/input/planes"

    # Add Shape node to get input shape
    shape_out_name = "__batch_shape__"
    shape_node = helper.make_node("Shape", [input_name], [shape_out_name],
                                  name="__get_input_shape__")
    # Add Gather to extract dim 0 (batch size) as a scalar, then Unsqueeze to [1]
    batch_idx_name = "__batch_gather_idx__"
    batch_idx_tensor = numpy_helper.from_array(
        np.array(0, dtype=np.int64), name=batch_idx_name)
    onnx_model.graph.initializer.append(batch_idx_tensor)

    batch_scalar_name = "__batch_scalar__"
    gather_node = helper.make_node("Gather", [shape_out_name, batch_idx_name],
                                   [batch_scalar_name], axis=0,
                                   name="__gather_batch__")

    unsqueeze_axes_name = "__unsqueeze_axes__"
    unsqueeze_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name=unsqueeze_axes_name)
    onnx_model.graph.initializer.append(unsqueeze_axes)

    batch_1d_name = "__batch_1d__"
    unsqueeze_node = helper.make_node("Unsqueeze",
                                      [batch_scalar_name, unsqueeze_axes_name],
                                      [batch_1d_name],
                                      name="__unsqueeze_batch__")

    # Insert these nodes at the beginning of the graph
    onnx_model.graph.node.insert(0, shape_node)
    onnx_model.graph.node.insert(1, gather_node)
    onnx_model.graph.node.insert(2, unsqueeze_node)

    # Build a map of initializer names for quick lookup
    init_map = {init.name: init for init in onnx_model.graph.initializer}

    # Now fix each Reshape node
    concat_counter = 0
    for node in list(onnx_model.graph.node):
        if node.op_type != "Reshape":
            continue
        shape_input_name = node.input[1]
        if shape_input_name not in init_map:
            continue  # shape is computed dynamically, skip
        shape_init = init_map[shape_input_name]
        shape_val = numpy_helper.to_array(shape_init).copy()
        if len(shape_val) < 1 or shape_val[0] != 1:
            continue  # batch dim is not 1, skip

        # Create a new constant for the tail dims (everything except batch)
        tail_dims = shape_val[1:]
        tail_name = f"__reshape_tail_{concat_counter}__"
        tail_tensor = numpy_helper.from_array(
            tail_dims.astype(np.int64), name=tail_name)
        onnx_model.graph.initializer.append(tail_tensor)

        # Create a Concat node: [batch_1d, tail_dims] -> new_shape
        new_shape_name = f"__reshape_dyn_shape_{concat_counter}__"
        concat_node = helper.make_node(
            "Concat",
            [batch_1d_name, tail_name],
            [new_shape_name],
            axis=0,
            name=f"__concat_reshape_{concat_counter}__"
        )

        # Insert concat node just before the current graph position
        # (we'll add all concat nodes after the unsqueeze)
        onnx_model.graph.node.insert(3 + concat_counter, concat_node)

        # Update the Reshape node to use the dynamic shape
        node.input[1] = new_shape_name

        concat_counter += 1

    print(f"  Fixed {concat_counter} Reshape nodes for dynamic batch support")

    # Also fix Expand nodes that may have hardcoded batch=1
    expand_fixes = 0
    for node in list(onnx_model.graph.node):
        if node.op_type != "Expand":
            continue
        shape_input_name = node.input[1]
        if shape_input_name not in init_map:
            continue
        shape_init = init_map[shape_input_name]
        shape_val = numpy_helper.to_array(shape_init).copy()
        if len(shape_val) < 1 or shape_val[0] != 1:
            continue

        tail_dims = shape_val[1:]
        tail_name = f"__expand_tail_{expand_fixes}__"
        tail_tensor = numpy_helper.from_array(
            tail_dims.astype(np.int64), name=tail_name)
        onnx_model.graph.initializer.append(tail_tensor)

        new_shape_name = f"__expand_dyn_shape_{expand_fixes}__"
        concat_node = helper.make_node(
            "Concat",
            [batch_1d_name, tail_name],
            [new_shape_name],
            axis=0,
            name=f"__concat_expand_{expand_fixes}__"
        )
        onnx_model.graph.node.insert(3 + concat_counter + expand_fixes, concat_node)
        node.input[1] = new_shape_name
        expand_fixes += 1

    if expand_fixes:
        print(f"  Fixed {expand_fixes} Expand nodes for dynamic batch support")

    onnx.save_model(
        onnx_model,
        output_path,
        save_as_external_data=False,  # Embed all weights in the .onnx file
    )

    # Clean up external data file if it exists
    ext_data_path = Path(output_path + ".data")
    if ext_data_path.exists():
        ext_data_path.unlink()
        print(f"  Cleaned up external data file: {ext_data_path}")

    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"ONNX model exported to: {output_path} ({file_size:.1f} MB)")


# =============================================================================
# Protobuf Wrapper (pure Python, no protoc needed)
# =============================================================================
# We write the lc0 Net protobuf message manually using raw wire format,
# since lc0 uses a custom protobuf compiler and we don't have the generated
# Python bindings.

def _varint_encode(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _field_varint(field_number: int, value: int) -> bytes:
    """Encode a varint field."""
    tag = (field_number << 3) | 0  # wire type 0 = varint
    return _varint_encode(tag) + _varint_encode(value)


def _field_fixed32(field_number: int, value: int) -> bytes:
    """Encode a fixed32 field."""
    tag = (field_number << 3) | 5  # wire type 5 = 32-bit
    return _varint_encode(tag) + struct.pack('<I', value)


def _field_bytes(field_number: int, data: bytes) -> bytes:
    """Encode a length-delimited field (bytes or string)."""
    tag = (field_number << 3) | 2  # wire type 2 = length-delimited
    return _varint_encode(tag) + _varint_encode(len(data)) + data


def _field_string(field_number: int, value: str) -> bytes:
    """Encode a string field."""
    return _field_bytes(field_number, value.encode('utf-8'))


def _field_submessage(field_number: int, data: bytes) -> bytes:
    """Encode a submessage field."""
    return _field_bytes(field_number, data)


def build_lc0_protobuf(onnx_bytes: bytes) -> bytes:
    """
    Build the lc0 Net protobuf message wrapping the ONNX model.

    Net {
      magic = 0x1c0                          # field 1, fixed32
      min_version = {major=0, minor=28}      # field 3, submessage
      format = {                             # field 4, submessage
        network_format = {                   # field 2, submessage
          input = INPUT_CLASSICAL_112_PLANE   # field 1, varint = 1
          output = OUTPUT_WDL                 # field 2, varint = 2
          network = NETWORK_ONNX             # field 3, varint = 5
          policy = POLICY_CLASSICAL          # field 4, varint = 1
          value = VALUE_WDL                  # field 5, varint = 2
          moves_left = MOVES_LEFT_NONE       # field 6, varint = 0
        }
      }
      onnx_model = {                         # field 11, submessage
        model = <onnx_bytes>                 # field 1, bytes
        data_type = FLOAT                    # field 2, varint = 1
        input_planes = "/input/planes"       # field 3, string
        output_wdl = "/output/wdl"           # field 5, string
        output_policy = "/output/policy"     # field 6, string
      }
    }
    """
    # Build EngineVersion (min_version): major=0 (field 1), minor=28 (field 2)
    min_version = _field_varint(1, 0) + _field_varint(2, 28)

    # Build NetworkFormat
    network_format = (
        _field_varint(1, 1) +   # input = INPUT_CLASSICAL_112_PLANE
        _field_varint(2, 2) +   # output = OUTPUT_WDL
        _field_varint(3, 5) +   # network = NETWORK_ONNX
        _field_varint(4, 1) +   # policy = POLICY_CLASSICAL
        _field_varint(5, 2) +   # value = VALUE_WDL
        _field_varint(6, 0)     # moves_left = MOVES_LEFT_NONE
    )

    # Build Format
    fmt = _field_submessage(2, network_format)  # network_format is field 2 of Format

    # Build OnnxModel
    onnx_model = (
        _field_bytes(1, onnx_bytes) +            # model
        _field_varint(2, 1) +                     # data_type = FLOAT
        _field_string(3, "/input/planes") +       # input_planes
        _field_string(5, "/output/wdl") +         # output_wdl
        _field_string(6, "/output/policy")        # output_policy
    )

    # Build Net
    net = (
        _field_fixed32(1, 0x1c0) +               # magic
        _field_submessage(3, min_version) +       # min_version
        _field_submessage(4, fmt) +               # format
        _field_submessage(11, onnx_model)         # onnx_model
    )

    return net


def write_lc0_weights(onnx_path: str, output_path: str) -> None:
    """Read an ONNX file and wrap it in lc0's .pb.gz format."""
    with open(onnx_path, 'rb') as f:
        onnx_bytes = f.read()

    net_bytes = build_lc0_protobuf(onnx_bytes)

    with gzip.open(output_path, 'wb') as f:
        f.write(net_bytes)

    print(f"lc0 weights written to: {output_path}")
    print(f"  ONNX model size: {len(onnx_bytes) / 1024 / 1024:.1f} MB")
    print(f"  Compressed size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export Marvin Chessformer to lc0-compatible .pb.gz format"
    )
    parser.add_argument(
        "--input", "-i", type=str, default="inference/marvin_large.pt",
        help="Path to Marvin .pt checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="marvin_for_leela.pb.gz",
        help="Output .pb.gz file path"
    )
    parser.add_argument(
        "--config", type=str, default="auto",
        help="Model config: auto, tiny, small, large"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (cpu recommended for ONNX export)"
    )
    parser.add_argument(
        "--onnx-opset", type=int, default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--model-py", type=str, default=None,
        help="Path to model.py (auto-detected from checkpoint dir)"
    )
    parser.add_argument(
        "--elo", type=int, default=None,
        help="Override DEFAULT_ELO for the exported model (default: 2300). "
             "Set to the target player's rating for fine-tuned models."
    )
    parser.add_argument(
        "--clock-seconds", type=float, default=None,
        help="Override DEFAULT_CLOCK_S for the exported model (default: 300.0)"
    )
    parser.add_argument(
        "--increment-seconds", type=float, default=None,
        help="Override DEFAULT_INC_S for the exported model (default: 0.0)"
    )
    args = parser.parse_args()

    # scripts/ sits one level below the repository root.  The earlier version
    # only inserted the scripts/ directory on sys.path, which meant
    # ``import inference`` failed because the package lives in the parent
    # directory.  Use parent.parent so the workspace root is added instead.
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint_path = Path(args.input)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path

    # Auto-detect model.py location
    model_py = args.model_py
    if model_py is None:
        model_py = repo_root / "model.py"
    model_py = Path(model_py)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model definition: {model_py}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()

    # Add repo root to path so ``inference`` and any sibling modules are
    # available; also allows importing model.py by relative path.
    sys.path.insert(0, str(repo_root))

    # Load Marvin model directly (inference.model_loader was removed; logic inlined here)
    import importlib.util

    def _load_model_module(path):
        spec = importlib.util.spec_from_file_location("marvin_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _detect_config(state):
        normalized = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        d_model = None
        for key, value in normalized.items():
            if "layers.0.attn.q_proj.weight" in key:
                d_model = value.shape[0]
                break
        if d_model is None:
            return "small"
        elo_only = any(
            "token_conditioning.token_pos_embedding.weight" in k and v.shape[0] == 2
            for k, v in normalized.items()
        )
        if d_model >= 550:
            return "large"
        elif d_model >= 400:
            return "small-notime" if elo_only else "small"
        else:
            return "tiny"

    module = _load_model_module(model_py)
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    config_name = args.config
    if config_name == "auto":
        config_name = _detect_config(state)
        print(f"Auto-detected model config: {config_name}")
    config_name = config_name.replace("-v2", "")

    _config_map = {
        "large":       module.CONFIG_LARGE,
        "tiny":        module.CONFIG_TINY,
        "small-notime": module.CONFIG_SMALL_NOTIME,
    }
    config = dict(_config_map.get(config_name, module.CONFIG_SMALL))
    model = module.Chessformer(config).to(device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded model: {config_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create adapter
    adapter = MarvinLeelaAdapter(model).to(args.device)
    
    # Apply CLI overrides for conditioning defaults
    if args.elo is not None:
        adapter.DEFAULT_ELO = args.elo
        print(f"[export] ELO override: {args.elo}")
    if args.clock_seconds is not None:
        adapter.DEFAULT_CLOCK_S = args.clock_seconds
        print(f"[export] Clock override: {args.clock_seconds}s")
    if args.increment_seconds is not None:
        adapter.DEFAULT_INC_S = args.increment_seconds
        print(f"[export] Increment override: {args.increment_seconds}s")

    # Quick sanity check
    print("Running sanity check...")
    with torch.no_grad():
        # Create a plausible input: mostly zeros (empty board) with plane 111 = all ones
        dummy = torch.zeros(1, 112, 8, 8, device=args.device)
        dummy[0, 111, :, :] = 1.0  # all-ones plane
        # Place white king on e1 (plane 5 = our king)
        dummy[0, 5, 0, 4] = 1.0
        # Place black king on e8 (plane 11 = their king)
        dummy[0, 11, 7, 4] = 1.0
        policy, wdl = adapter(dummy)
        print(f"  Policy shape: {policy.shape} (expected: [1, 1858])")
        print(f"  WDL shape: {wdl.shape} (expected: [1, 3])")
        print(f"  WDL sum: {wdl.sum().item():.4f} (expected: ~1.0)")
        print(f"  Policy range: [{policy.min().item():.4f}, {policy.max().item():.4f}]")
    print()

    # Export to ONNX
    onnx_path = args.output.replace('.pb.gz', '.onnx')
    print(f"Exporting to ONNX: {onnx_path}")
    export_to_onnx(adapter, onnx_path, opset_version=args.onnx_opset)
    print()

    # Wrap in lc0 protobuf
    print(f"Wrapping in lc0 protobuf: {args.output}")
    write_lc0_weights(onnx_path, args.output)
    print()

    print("=" * 60)
    print("DONE! To use with lc0:")
    print(f"  ./lc0 --weights={args.output} --backend=onnx-cpu")
    print()
    print("Or with CUDA:")
    print(f"  ./lc0 --weights={args.output} --backend=onnx-cuda")
    print()
    print("IMPORTANT NOTES:")
    print("  - Input format is INPUT_CLASSICAL_112_PLANE (no board canonicalization)")
    print("  - En passant info is NOT available in classical format")
    print("    (for EP support, need to modify for castling-plane format)")
    print(f"  - ELO is set to {adapter.DEFAULT_ELO}, clock to {adapter.DEFAULT_CLOCK_S:.0f}+{adapter.DEFAULT_INC_S:.0f}")
    print("  - Use --elo, --clock-seconds, --increment-seconds to change")
    print("  - Time history, ply are hardcoded defaults")
    print("  - The model's time/style heads are not exposed to lc0")
    print("=" * 60)


if __name__ == "__main__":
    main()
