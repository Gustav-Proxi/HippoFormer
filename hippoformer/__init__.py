"""
HippoFormer: Hippocampal Memory Selection for Transformers

A novel memory-augmented transformer architecture that implements
biologically-inspired memory mechanisms:
- Salience Gate: Dual-pathway learned importance scoring (SPW-R analogue)
- Memory Consolidator: Priority-based replay with exponential decay
- Drift Calibrator: Stable retrieval via affine correction

Paper: "HippoFormer: Hippocampal Memory Selection for Transformers"
Based on: "Selection of experience for memory by hippocampal sharp wave ripples" (Yang et al., 2024)
"""

from hippoformer.config import HippoFormerConfig
from hippoformer.model import HippoFormer
from hippoformer.salience.gate import SalienceGate
from hippoformer.memory.buffer import DifferentiablePriorityBuffer
from hippoformer.drift.calibrator import EmbeddingDriftCalibrator
from hippoformer.losses import HippoFormerLoss, ConsolidationRScore

__all__ = [
    "HippoFormerConfig",
    "HippoFormer",
    "SalienceGate",
    "DifferentiablePriorityBuffer",
    "EmbeddingDriftCalibrator",
    "HippoFormerLoss",
    "ConsolidationRScore",
]
__version__ = "0.1.0"
