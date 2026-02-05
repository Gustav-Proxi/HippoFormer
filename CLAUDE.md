# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dev Notes
`dev-notes.md` is a running log of codebase state and changes. It is kept up to date as the project evolves.

## Commands

```bash
# install (editable + dev deps)
pip install -e ".[dev]"

# install with training dependencies
pip install -e ".[train]"

# install with evaluation dependencies
pip install -e ".[eval]"

# install everything
pip install -e ".[all]"

# run all tests
pytest

# run hippoformer tests
pytest tests/test_hippoformer.py -v

# run evaluation tests
pytest tests/test_evaluation.py -v

# run evaluation pipeline
python -m evaluation.runner --datasets wikitext-2 --max-samples 100

# benchmark training speed
python scripts/benchmark_training.py
```

## Architecture

HippoFormer: Hippocampal Memory Selection for Transformers

An integrated transformer architecture that embeds hippocampal memory mechanisms
directly into the model. All hyperparameters live in `hippoformer/config.py`.

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│           Base Transformer (Llama/Gemma, frozen+LoRA)       │
└─────────────────────────────────────────────────────────────┘
     │ hidden states h_t
     ▼
┌─────────────────────────────────────────────────────────────┐
│  SALIENCE GATE (hippoformer/salience/gate.py)               │
│  • Local pathway: token-intrinsic importance (MLP)          │
│  • Global pathway: contextual importance (cross-attn)       │
│  • Output: salience scores [0,1], importance weights [1,5]  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  DRIFT CALIBRATOR (hippoformer/drift/calibrator.py)         │
│  • Anchor embeddings from training                          │
│  • Learned affine correction: h' = Ah + b                   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  MEMORY CONSOLIDATOR (hippoformer/memory/buffer.py)         │
│  • Priority buffer: (keys, values, priorities, ages)        │
│  • Multi-round replay consolidation with exponential decay  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT FUSION (hippoformer/model.py)                       │
│  • Cross-attention: Q=hidden, K/V=consolidated memory       │
│  • Gated fusion: out = g*memory + (1-g)*hidden              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Output Logits
```

### Key conventions

- `hippoformer/config.py` is the single source of truth for all hyperparameters
- Importance weights range from [1.0, 5.0]: 1.0 for untagged, 2.0-5.0 for tagged
- Decay rate (0.9) and max replay rounds (10) control consolidation
- The architecture wraps any HuggingFace causal LM (Gemma, Llama, etc.)

### File structure

```
hippoformer/
├── config.py           # HippoFormerConfig dataclass
├── model.py            # Main HippoFormer model
├── losses.py           # Multi-objective loss functions
├── train.py            # Training script
├── salience/
│   └── gate.py         # SalienceGate module
├── memory/
│   └── buffer.py       # DifferentiablePriorityBuffer
└── drift/
    └── calibrator.py   # EmbeddingDriftCalibrator

evaluation/
├── __init__.py         # Public API
├── metrics.py          # Perplexity, BLEU, ROUGE, F1, exact match
├── datasets.py         # WikiText-103, PG-19, NarrativeQA loaders
├── ablation.py         # Ablation framework (15 variants)
├── visualization.py    # Salience heatmaps, training curves
├── statistics.py       # Significance tests, confidence intervals
└── runner.py           # CLI evaluation pipeline

scripts/
└── benchmark_training.py  # Hardware speed benchmark
```

### Evaluation

The `evaluation/` package provides complete infrastructure for paper publication:

- **Metrics**: perplexity (standard + by-position), BLEU, ROUGE, F1, exact match
- **Datasets**: WikiText-2/103, PG-19, NarrativeQA, SCROLLS
- **Ablation**: 15 pre-defined variants with multi-seed support
- **Statistics**: paired/unpaired tests, multiple comparison correction
- **Visualization**: salience heatmaps, training curves, ablation plots

### Hardware (tested on M4 MacBook Air 16GB)

- Gemma-2B + LoRA fits in memory (~8-10GB peak)
- Training speed: ~3-5s/step on MPS
- WikiText-2 full epoch: ~15-25 hours locally
- Quick test (1K samples): ~30 minutes
