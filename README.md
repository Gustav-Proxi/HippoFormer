<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.36+-yellow?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
</p>

<h1 align="center">HippoFormer</h1>
<h3 align="center">Hippocampal Memory Selection for Transformers</h3>

<p align="center">
  <em>A biologically-inspired memory architecture that brings hippocampal memory consolidation to large language models</em>
</p>

<p align="center">
  <a href="https://huggingface.co/Gustav-Proxi/HippoFormer-Gemma2B">Model</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#citation">Citation</a>
</p>

---

## Overview

**HippoFormer** integrates hippocampal memory mechanisms directly into transformer architectures. Inspired by how the human hippocampus selectively consolidates important memories through Sharp Wave Ripples (SPW-Rs), HippoFormer learns to:

- **Selectively tag** important tokens (like the brain identifies significant events)
- **Consolidate memories** through priority-based replay (like sleep consolidation)
- **Maintain stable representations** through drift calibration

<p align="center">
  <img src="docs/architecture.png" alt="HippoFormer Architecture" width="700">
</p>

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│           Base Transformer (Gemma-2B, frozen + LoRA)        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  SALIENCE GATE                                              │
│  Dual-pathway importance scoring inspired by SPW-Rs         │
│  • Local: token-intrinsic importance (single neuron)        │
│  • Global: contextual importance (population synchrony)     │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  MEMORY CONSOLIDATOR                                        │
│  Priority buffer with multi-round replay                    │
│  • Exponential decay mimics memory consolidation            │
│  • High-importance tokens persist longer                    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT FUSION                                              │
│  Cross-attention integration of consolidated memories       │
│  • Gated fusion: output = g·memory + (1-g)·hidden          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Output Logits
```

---

## Results

### Perplexity Comparison

| Model | Parameters | WikiText-2 PPL |
|-------|------------|----------------|
| GPT-2 | 124M | 29.41 |
| Gemma-2B | 2B | ~18 |
| **HippoFormer** | 2B + 15M | **11.83** |

### Ablation Study

Our ablation analysis validates that **both** hippocampal components are essential:

| Configuration | PPL | Δ PPL |
|--------------|-----|-------|
| Full HippoFormer | 11.83 | — |
| Without Salience Gate | 39.75 | +27.92 |
| Without Memory Buffer | 89.84 | +78.01 |
| Random Salience | 89.84 | +78.01 |

### Brain-Like Behavior Validation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Content/Function Word Ratio | **2.11x** | Content words tagged more (selective memory) |
| Long-Range PPL Benefit | **+6.95** | Better on late tokens (remembers context) |
| Buffer Priority | **4.9/5.0** | High-importance items retained |
| Temporal Coherence | **0.58** | Nearby tokens tagged together |

---

## Installation

```bash
# Clone repository
git clone https://github.com/Gustav-Proxi/HippoFormer.git
cd HippoFormer

# Install with training dependencies
pip install -e ".[train]"

# Or install everything
pip install -e ".[all]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- CUDA 11.8+ (for GPU training)

---

## Usage

### Quick Start

```python
from hippoformer import HippoFormer, HippoFormerConfig
from transformers import AutoTokenizer
import torch

# Initialize
config = HippoFormerConfig(
    base_model_name="google/gemma-2b",
    freeze_base=True,
    use_lora=True,
)
model = HippoFormer(config)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Load pretrained weights
ckpt = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)

# Generate
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

### Load from HuggingFace

```python
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download(
    repo_id="Gustav-Proxi/HippoFormer-Gemma2B",
    filename="pytorch_model.pt"
)

# Load
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)
```

### Training

```bash
# Train on WikiText-2
python -m hippoformer.train \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --batch_size 8 \
    --num_epochs 3 \
    --output_dir ./outputs
```

### Evaluation

```bash
# Run comprehensive evaluation
python -m evaluation.comprehensive_eval \
    --checkpoint ./outputs/checkpoint-step-110000/checkpoint.pt \
    --output results.json \
    --device cuda
```

---

## Architecture Details

### Salience Gate

The salience gate implements a dual-pathway importance scoring mechanism inspired by hippocampal Sharp Wave Ripples:

```python
# Local pathway: token-intrinsic importance
local_scores = MLP(hidden_states)  # Like single-electrode ripple detection

# Global pathway: contextual importance
global_scores = CrossAttention(hidden_states)  # Like population synchrony

# Combined with learnable weighting
salience = sigmoid(w * local + (1-w) * global - threshold)
```

### Memory Consolidator

Priority-based buffer with multi-round replay consolidation:

```python
# Store with priority = salience * importance_weight
buffer.store(keys, values, priorities)

# Multi-round replay with exponential decay
for round in range(max_rounds):
    consolidated = replay(buffer, decay_rate ** round)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | 2048 | Memory buffer capacity |
| `decay_rate` | 0.9 | Consolidation decay per round |
| `importance_range` | [1.0, 5.0] | Min/max importance weights |
| `salience_threshold` | 0.0 | Initial threshold (learned) |

---

## Neuroscience Background

HippoFormer is inspired by hippocampal memory consolidation mechanisms:

| Brain Mechanism | HippoFormer Implementation |
|-----------------|---------------------------|
| Sharp Wave Ripples (SPW-Rs) | Salience Gate (dual-pathway detection) |
| Memory tagging | Importance weights [1.0 - 5.0] |
| Sleep replay | Multi-round consolidation with decay |
| Synaptic homeostasis | Drift calibration |

**Key insight:** The hippocampus doesn't remember everything equally. It selectively tags important experiences and consolidates them through replay during sleep. HippoFormer brings this mechanism to transformers.

---

## Project Structure

```
HippoFormer/
├── hippoformer/
│   ├── config.py           # HippoFormerConfig
│   ├── model.py            # Main HippoFormer model
│   ├── train.py            # Training script
│   ├── losses.py           # Multi-objective losses
│   ├── salience/
│   │   └── gate.py         # SalienceGate module
│   ├── memory/
│   │   └── buffer.py       # DifferentiablePriorityBuffer
│   └── drift/
│       └── calibrator.py   # EmbeddingDriftCalibrator
├── evaluation/
│   ├── metrics.py          # PPL, BLEU, ROUGE, F1
│   ├── ablation.py         # Ablation framework
│   ├── comprehensive_eval.py  # Full evaluation suite
│   └── visualization.py    # Paper figures
├── scripts/
│   ├── runpod/             # Cloud training scripts
│   └── aws/                # AWS deployment
└── tests/                  # Unit tests
```

---

## Citation

```bibtex
@misc{hippoformer2025,
  title={HippoFormer: Hippocampal Memory Selection for Transformers},
  author={Gustav-Proxi},
  year={2025},
  howpublished={\url{https://github.com/Gustav-Proxi/HippoFormer}},
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built on [Gemma](https://ai.google.dev/gemma) by Google DeepMind
- Inspired by hippocampal memory research
- Training infrastructure on [RunPod](https://runpod.io)

---

<p align="center">
  <strong>HippoFormer</strong> — Bringing biological memory to artificial intelligence
</p>
