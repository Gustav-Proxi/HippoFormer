# HippoFormer Collaboration Guide for Sanika

Welcome to the HippoFormer project! This guide explains the codebase, what's been done, and where you can contribute.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What's Already Done](#whats-already-done)
3. [Codebase Structure](#codebase-structure)
4. [Code Contributions](#code-contributions-for-sanika)
5. [Paper Contributions](#paper-contributions-for-sanika)
6. [Contribution Split](#contribution-split)
7. [Getting Started](#getting-started)

---

## Project Overview

**HippoFormer** is a transformer architecture inspired by how the human hippocampus processes and consolidates memories. Instead of treating all tokens equally, it:

1. **Selectively tags important tokens** (like how the brain marks significant events)
2. **Consolidates memories through replay** (like sleep consolidation)
3. **Maintains stable representations** (like synaptic homeostasis)

### The Core Idea

The hippocampus doesn't remember everything - it selectively consolidates important memories during sleep through "Sharp Wave Ripples" (SPW-Rs). We brought this mechanism to transformers.

### Results Achieved

| Metric | Value |
|--------|-------|
| Perplexity (WikiText-2) | **11.83** (vs 18 for base Gemma-2B) |
| Content/Function word ratio | 2.11x (selective memory works!) |
| Training time | ~24 hours on RTX 4090 |

---

## What's Already Done

### Completed by Vaishak:

| Component | Status | Description |
|-----------|--------|-------------|
| Core Architecture | ✅ Done | SalienceGate, MemoryBuffer, DriftCalibrator |
| Model Integration | ✅ Done | HippoFormer wrapper for HuggingFace models |
| Training Pipeline | ✅ Done | Full training script with LoRA, checkpointing |
| Model Training | ✅ Done | Trained on WikiText-2, best checkpoint at step-110k |
| Basic Evaluation | ✅ Done | Perplexity evaluation |
| Ablation Framework | ✅ Done | 15 ablation variants |
| Comprehensive Eval | ✅ Done | Brain-like behavior validation |
| HuggingFace Upload | ✅ Done | Model available on HF Hub |
| GitHub Repo | ✅ Done | Public repo with README |
| Paper Figures | ✅ Done | Architecture, ablation, salience heatmaps |

---

## Codebase Structure

```
HippoFormer/
│
├── hippoformer/                 # CORE MODEL CODE
│   ├── __init__.py
│   ├── config.py               # All hyperparameters (IMPORTANT - read this first!)
│   ├── model.py                # Main HippoFormer class
│   ├── losses.py               # Loss functions
│   ├── train.py                # Training script
│   │
│   ├── salience/
│   │   └── gate.py             # SalienceGate - dual pathway importance scoring
│   │                           # This is the "Sharp Wave Ripple" detector
│   │
│   ├── memory/
│   │   └── buffer.py           # DifferentiablePriorityBuffer
│   │                           # This is the "sleep replay" mechanism
│   │
│   └── drift/
│       └── calibrator.py       # EmbeddingDriftCalibrator
│                               # Keeps embeddings stable over time
│
├── evaluation/                  # EVALUATION CODE
│   ├── __init__.py
│   ├── metrics.py              # Perplexity, BLEU, ROUGE, F1
│   ├── datasets.py             # Dataset loaders
│   ├── ablation.py             # Ablation study framework
│   ├── visualization.py        # Plotting functions
│   ├── statistics.py           # Statistical tests
│   ├── comprehensive_eval.py   # Full evaluation suite
│   └── runner.py               # CLI for running evaluations
│
├── docs/
│   └── figures/                # Paper figures (PNG + PDF)
│       ├── architecture.png
│       ├── ablation_study.png
│       ├── salience_heatmap.png
│       ├── perplexity_position.png
│       ├── brain_mapping.png
│       └── results_summary.png
│
├── scripts/
│   ├── generate_paper_figures.py
│   └── runpod/                 # Cloud training scripts
│
├── tests/                      # Unit tests
│
├── README.md                   # Project README
├── dev-notes.md               # Development log
└── CLAUDE.md                  # AI assistant instructions
```

---

## Code Contributions for Sanika

### Priority 1: High Impact, Moderate Difficulty

#### 1. Long-Context Evaluation (PG-19, NarrativeQA)
**File to modify:** `evaluation/runner.py`, `evaluation/datasets.py`

**What to do:**
- Run evaluation on PG-19 dataset (very long documents)
- Run evaluation on NarrativeQA (question answering over stories)
- This will prove HippoFormer's memory advantage on long sequences

**How:**
```bash
# After setting up, run:
python -m evaluation.runner --datasets pg19 narrativeqa --checkpoint <path>
```

**Why it matters:** Our main claim is that HippoFormer handles long-range dependencies better. We need long-context benchmarks to prove this.

---

#### 2. More Ablation Experiments
**File:** `evaluation/ablation.py`

**What to do:**
- Run ablations with multiple random seeds (3 seeds minimum)
- Test different buffer sizes: 512, 1024, 2048, 4096
- Test different decay rates: 0.8, 0.85, 0.9, 0.95

**How:**
```python
from evaluation.ablation import AblationRunner

runner = AblationRunner(base_checkpoint="path/to/checkpoint.pt")
results = runner.run_ablation_suite(
    variants=["buffer_512", "buffer_1024", "buffer_4096", "decay_0.8", "decay_0.95"],
    seeds=[42, 123, 456]
)
```

---

#### 3. Statistical Significance Tests
**File:** `evaluation/statistics.py`

**What to do:**
- Run paired t-tests comparing HippoFormer vs baselines
- Calculate 95% confidence intervals
- Apply multiple comparison correction (Bonferroni)

**Code already exists, just needs to be run:**
```python
from evaluation.statistics import compare_models, create_results_table

comparison = compare_models(
    hippoformer_results,
    baseline_results,
    metrics=["perplexity", "bleu", "rouge"]
)
table = create_results_table(comparison)
```

---

### Priority 2: Medium Impact, Lower Difficulty

#### 4. Improve Visualizations
**File:** `evaluation/visualization.py`, `scripts/generate_paper_figures.py`

**What to do:**
- Add error bars to ablation plots
- Create attention pattern visualizations
- Add more salience heatmap examples with different text types

---

#### 5. Add More Metrics
**File:** `evaluation/metrics.py`

**What to do:**
- Add BERTScore metric
- Add MAUVE score for generation quality
- Add bits-per-byte metric

---

#### 6. Unit Tests
**File:** `tests/`

**What to do:**
- Add more edge case tests
- Add integration tests
- Improve test coverage

---

### Priority 3: Advanced (If Time Permits)

#### 7. Try Different Base Models
- Test with Llama-2-7B
- Test with Mistral-7B
- Compare memory advantage across model sizes

#### 8. Efficiency Optimizations
- Profile memory usage
- Optimize buffer operations
- Add gradient checkpointing

---

## Paper Contributions for Sanika

### Paper Structure

```
1. Abstract                          [Vaishak - Done]
2. Introduction                      [Vaishak - Draft, Sanika - Review/Edit]
3. Related Work                      [SANIKA - Primary Author]
4. Method                            [Vaishak - Done]
5. Experiments                       [SPLIT]
   5.1 Setup                         [Vaishak]
   5.2 Main Results                  [Vaishak]
   5.3 Ablation Study               [SANIKA - Run & Write]
   5.4 Long-Context Evaluation      [SANIKA - Run & Write]
   5.5 Brain-Like Behavior Analysis [SPLIT]
6. Discussion                        [SPLIT]
7. Conclusion                        [Vaishak - Draft, Sanika - Review]
8. References                        [SANIKA - Format & Verify]
```

---

### Section 3: Related Work (SANIKA - Primary Author)

**What to cover:**

#### 3.1 Memory-Augmented Transformers
- Memorizing Transformers (Wu et al., 2022)
- Longformer (Beltagy et al., 2020)
- BigBird (Zaheer et al., 2020)
- Transformer-XL (Dai et al., 2019)

**Key point:** These use static/retrieval-based memory. HippoFormer uses *learned* importance and *consolidation*.

#### 3.2 Neuroscience-Inspired AI
- Complementary Learning Systems (McClelland et al., 1995)
- Memory replay in RL (Mnih et al., 2015 - DQN)
- Sleep and memory consolidation literature

**Key point:** We're the first to bring SPW-R mechanisms to transformers.

#### 3.3 Efficient Attention
- Sparse attention patterns
- Linear attention variants
- State space models (Mamba, etc.)

**Key point:** Our approach is orthogonal - we focus on *what* to remember, not *how* to compute attention.

---

### Section 5.3: Ablation Study (SANIKA - Run & Write)

**What to do:**
1. Run ablations with 3 random seeds
2. Create table with mean ± std
3. Write analysis explaining why each component matters

**Template:**
```markdown
| Variant | PPL (mean ± std) | Δ PPL |
|---------|------------------|-------|
| Full Model | 11.83 ± 0.2 | - |
| No Salience | 39.75 ± 1.1 | +27.9 |
| No Memory | 89.84 ± 2.3 | +78.0 |
| Random Salience | 89.84 ± 2.1 | +78.0 |
| Buffer 512 | ? | ? |
| Buffer 4096 | ? | ? |
| Decay 0.8 | ? | ? |
| Decay 0.95 | ? | ? |
```

---

### Section 5.4: Long-Context Evaluation (SANIKA - Run & Write)

**What to do:**
1. Run PG-19 evaluation
2. Run NarrativeQA evaluation
3. Create perplexity-by-position plot
4. Write analysis of long-range benefits

**Expected narrative:**
- HippoFormer maintains lower perplexity at later positions
- Memory buffer retains important context from earlier in the sequence
- Quantify the "memory advantage" gap

---

### Section 5.5: Brain-Like Behavior (SPLIT)

**Sanika's part:**
- Literature review connecting results to neuroscience
- Statistical analysis of content vs function word tagging
- Discussion of biological plausibility

**What we found:**
- Content words tagged 2.11x more than function words (like hippocampus!)
- Temporal coherence of 0.58 (nearby tokens tagged together)
- High-priority items retained in buffer (4.9/5.0 mean priority)

---

### References (SANIKA - Format & Verify)

**What to do:**
1. Compile all citations in BibTeX format
2. Verify all citations are accurate
3. Ensure consistent formatting
4. Check for missing important papers

---

## Contribution Split

### Overall Split: Vaishak 70% / Sanika 30%

| Category | Vaishak | Sanika |
|----------|---------|--------|
| **Idea & Architecture** | 100% | 0% |
| **Core Implementation** | 100% | 0% |
| **Training** | 100% | 0% |
| **Basic Evaluation** | 100% | 0% |
| **Extended Evaluation** | 30% | 70% |
| **Paper Writing** | 60% | 40% |
| **Literature Review** | 20% | 80% |
| **Figures & Tables** | 70% | 30% |

### Author Order
**Vaishak Girish Kumar, Sanika**

(First author did core work; second author contributed to evaluation and writing)

---

## Getting Started

### 1. Clone and Setup

```bash
git clone https://github.com/Gustav-Proxi/HippoFormer.git
cd HippoFormer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[all]"
```

### 2. Download Checkpoint

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="Gustav-Proxi/HippoFormer-Gemma2B",
    filename="pytorch_model.pt"
)
print(f"Checkpoint at: {ckpt_path}")
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Run Quick Evaluation

```bash
python -m evaluation.runner --datasets wikitext-2 --max-samples 100
```

### 5. Read the Code

**Start here:**
1. `hippoformer/config.py` - All hyperparameters
2. `hippoformer/model.py` - Main model class
3. `hippoformer/salience/gate.py` - The core innovation
4. `README.md` - Project overview

---

## Communication

- **Questions?** Create a GitHub issue or message directly
- **Code changes?** Create a branch, make changes, open a PR
- **Paper drafts?** Share via Google Docs or Overleaf

---

## Timeline Suggestion

| Week | Task |
|------|------|
| Week 1 | Setup, read code, run basic evaluation |
| Week 2 | Run long-context experiments (PG-19) |
| Week 3 | Run ablation experiments with seeds |
| Week 4 | Write Related Work section |
| Week 5 | Write Ablation & Long-Context sections |
| Week 6 | Review, polish, finalize paper |

---

## Questions to Discuss

1. Which conference/journal are we targeting?
2. What's the deadline?
3. Do we need more baselines?
4. Should we try a different base model?

---

Good luck! The hard part (building the model) is done. Now we need to properly evaluate it and write it up. Your contribution to the evaluation and paper is crucial for publication.

**- Vaishak**
