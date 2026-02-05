# Dev Notes

Running log of codebase state and changes. Updated by Claude as the project evolves.

---

## 2026-02-04

### Initial implementation complete

Tech stack: Python 3.10+, numpy, scipy, scikit-learn, umap-learn, pytest.

**Spec files**
- `brain.md` – full neural-encoder + tagging + prompt architecture spec.
- `evaluation method.md` – dual-tier eval framework with metrics & validation.

**Core pipeline (`brain/`)**
- `config.py`   – all shared constants (ripple band, thresholds, component counts…).
- `encoder.py`  – `SeqNMF` (KL multiplicative updates with temporal convolution),
                  `DriftCalibrator` (affine recalibration), `NeuralEncoder` (orchestrator).
- `tagger.py`   – `SPWRTagger`: bandpass + Hilbert envelope detection of SPW-Rs,
                  event tagging, importance-weight scaling [2–5] by ripple amplitude.
- `decoder.py`  – `kNNDecoder` (sklearn KNN + CV + binomial p-value),
                  `NeuralTokenizer` (k-means centroid quantisation → `<N_STATE_i>` tokens
                  + system/instruction prompt builder).
- `replay.py`   – `ReplayEngine`: priority replay with exponential weight decay.

**Evaluation (`evaluation/`)**
- `tier1_neural.py`  – confusion matrix, Poisson + shuffled null models, within-event coherence.
- `tier2_llm.py`     – consolidation R-score (target ≥ 0.86), faithfulness, hallucination rate,
                        latency benchmark (target < 20 ms).
- `diagnostics.py`   – mean trial error, trajectory length, LOTO validation,
                        neuron-downsampling stress test.

**Tests (`tests/`)**
- `test_encoder.py`    – SeqNMF shapes/signs/convergence, DriftCalibrator correction, NeuralEncoder round-trip.
- `test_tagger.py`     – SPW-R detection (injected ripples + noise FP rate), tagging correctness, weight ordering.
- `test_decoder.py`    – kNN accuracy on separated clusters, tokenizer format + nearest-centroid, prompt content.
- `test_replay.py`     – decay termination, round counts, consolidated ordering.
- `test_evaluation.py` – all eval functions: confusion matrix, coherence, consolidation, faithfulness,
                          hallucination, latency, trial error, trajectory, LOTO.

**Entry point**
- `main.py` – synthetic-data pipeline demo; runs all 6 stages and prints a summary.

### Pending / next steps (original pipeline)
- No real BCI data ingestion yet (main.py uses synthetic Poisson spikes).
- `LLMRunner` wrapper for local quantised SLM (Gemma 3 270M / SmolLM2 360M) not yet implemented.
- Soft-prompt linear projection layer is described in spec but stubbed out.
- `null_model_test` in tier1 is expensive (100 kNN CV loops × 2 nulls); consider
  reducing `n_trials` or parallelising for large datasets.

---

## 2026-02-04 (later)

### HippoFormer: Integrated Architecture

Major pivot from external BCI pipeline to **integrated transformer architecture** for pure NLP.
The hippocampal memory mechanisms are now built into the model, not external preprocessing.

**New package: `hippoformer/`**

Architecture:
```
Input Tokens → Base LLM (frozen+LoRA) → Hidden States
    → SalienceGate (dual-pathway learned importance)
    → EmbeddingDriftCalibrator (affine correction)
    → DifferentiablePriorityBuffer (multi-round consolidation)
    → OutputFusion (cross-attention + gating)
    → Output Logits
```

**Core modules:**
- `hippoformer/config.py` - HippoFormerConfig dataclass
- `hippoformer/salience/gate.py` - SalienceGate: dual local+global pathway importance scoring
- `hippoformer/memory/buffer.py` - DifferentiablePriorityBuffer: priority-based replay
- `hippoformer/drift/calibrator.py` - EmbeddingDriftCalibrator: affine drift correction
- `hippoformer/model.py` - HippoFormer: main model wrapping any HuggingFace LLM
- `hippoformer/losses.py` - Multi-objective loss (LM + salience-weighted + sparsity)
- `hippoformer/train.py` - Training script with HuggingFace integration

**Mapping from original pipeline:**
| Original (external) | HippoFormer (integrated) |
|---------------------|--------------------------|
| SPWRTagger (fixed bandpass+threshold) | SalienceGate (learned dual-pathway) |
| ReplayEngine (fixed decay) | DifferentiablePriorityBuffer (learned consolidation) |
| DriftCalibrator (least-squares) | EmbeddingDriftCalibrator (learned affine) |
| NeuralTokenizer | Removed (standard tokenization) |
| SeqNMF/UMAP | Removed (use transformer embeddings) |

**Key novelties vs. prior work:**
1. Dual-pathway salience (local+global) vs. Titans' single surprise
2. Multi-round replay with decay vs. Memorizing Transformers' static retrieval
3. Explicit importance weights [1.0-5.0] from neuroscience
4. Drift calibration (unique - no existing work addresses this)
5. End-to-end differentiable

**Dependencies added:**
- torch>=2.0
- transformers>=4.36
- peft>=0.7 (optional, for LoRA)
- datasets>=2.14 (optional, for training)
- accelerate>=0.24 (optional, for training)

**Install:**
```bash
pip install -e ".[train]"  # with training deps
pip install -e ".[all]"    # everything
```

### Pending / next steps (HippoFormer)
- ~~Unit tests for hippoformer modules~~ ✅ Done
- Benchmark on WikiText-2 perplexity
- ~~Ablation studies (remove each component)~~ ✅ Framework built
- Long-context evaluation (NarrativeQA, QuALITY)

---

## 2026-02-05

### Evaluation Infrastructure Complete

Built comprehensive evaluation suite for research paper publication.

**New package: `evaluation/`**

```
evaluation/
├── __init__.py          # Public API exports
├── metrics.py           # Perplexity, BLEU, ROUGE, F1, exact match
├── datasets.py          # WikiText-103, PG-19, NarrativeQA, SCROLLS loaders
├── ablation.py          # 15 ablation variants, component isolation
├── visualization.py     # Salience heatmaps, training curves, ablation plots
├── statistics.py        # Significance tests, CI, multiple comparison correction
└── runner.py            # CLI orchestrator with full pipeline
```

**Metrics implemented (`metrics.py`):**
- `compute_perplexity()` - standard + by-position analysis
- `compute_bleu()` - BLEU-1/2/3/4 with smoothing
- `compute_rouge()` - ROUGE-1/2/L F1 scores
- `compute_f1()` - token-level F1 for QA
- `compute_exact_match()` - with normalization
- `compute_hippoformer_metrics()` - salience, buffer utilization
- `compute_generation_quality()` - generation evaluation

**Dataset loaders (`datasets.py`):**
- WikiText-2/103 (standard LM benchmarks)
- PG-19 (long-context, streaming with chunking)
- NarrativeQA (story comprehension QA)
- SCROLLS suite (qasper, quality, etc.)
- Factory function `create_eval_dataloader()` for unified interface

**Ablation framework (`ablation.py`):**
- 15 pre-defined ablation variants:
  - Component removal: no_salience, no_memory, no_drift
  - Salience ablations: random, fixed (0.5)
  - Buffer size: 512, 1024, 4096
  - Decay rate: 0.8, 0.95, 0.99
  - Importance range: [1,3], [1,10]
  - Baselines: base_model_only, lora_only
- `AblationRunner` class for systematic experiments
- Multi-seed support with result aggregation

**Visualization (`visualization.py`):**
- `plot_salience_heatmap()` - token-level salience visualization
- `plot_memory_dynamics()` - buffer utilization over time
- `plot_training_curves()` - loss curves with smoothing
- `plot_ablation_comparison()` - bar charts with error bars
- `plot_perplexity_by_position()` - long-range dependency analysis
- `create_paper_figures()` - batch generate all paper figures

**Statistical testing (`statistics.py`):**
- `compute_confidence_interval()` - t-distribution and bootstrap
- `paired_significance_test()` - t-test or Wilcoxon (auto-select)
- `unpaired_significance_test()` - independent samples
- `multiple_comparison_correction()` - Bonferroni, Holm, FDR-BH
- `aggregate_seeds()` - mean/std/CI across random seeds
- `compare_models()` - systematic baseline comparison
- `create_results_table()` - markdown table generation

**CLI runner (`runner.py`):**
```bash
# Quick evaluation
python -m evaluation.runner --datasets wikitext-2 --max-samples 100

# Full evaluation with figures
python -m evaluation.runner \
    --checkpoint ./hippoformer_output/checkpoint-epoch-0 \
    --datasets wikitext-2 wikitext-103 \
    --run-position-analysis \
    --output-dir ./eval_results
```

**Tests (`tests/test_evaluation.py`):**
- 24 tests covering all modules
- All passing

**Dependencies added to `pyproject.toml`:**
```toml
[project.optional-dependencies]
eval = [
    "datasets>=2.14",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scipy>=1.10",
]
```

### Hardware Benchmark (M4 MacBook Air, 16GB)

Ran `scripts/benchmark_training.py` to estimate training time.

**Measured (GPT-2 124M baseline):**
- Device: Apple MPS
- Step time: 0.56s (batch=2, seq=512)
- Tokens/second: ~1,800

**Estimated for Gemma-2B + LoRA:**
| Scenario | Samples | Time |
|----------|---------|------|
| Quick test | 1,000 | ~30 min |
| Dev iteration | 5,000 | ~2-3 hours |
| Full WikiText-2 | 36,000 | ~15-25 hours |
| Cloud A100 | 36,000 | ~1-2 hours |

**Memory:** ~8-10GB peak for Gemma-2B with LoRA (fits in 16GB)

### Recommended Paper Experiment Plan

**Tier 1: Validation (local Mac)**
- WikiText-2 perplexity baseline
- Training convergence verification
- Component unit tests

**Tier 2: Core Results (local or cloud)**
- Ablation study (15 variants × 3 seeds)
- WikiText-103 perplexity comparison
- Salience visualization analysis

**Tier 3: Memory Advantage (cloud recommended)**
- PG-19 long-context evaluation
- NarrativeQA downstream task
- Perplexity-by-position analysis

**Tier 4: Paper Polish**
- Statistical significance tests
- Generate publication figures
- Results tables with confidence intervals

### AWS Infrastructure

Added cloud deployment scripts for consistent hardware benchmarking.

**New files:**
- `scripts/aws/setup.sh` - EC2 instance bootstrap (installs deps, clones repo, verifies GPU)
- `scripts/aws/run_training.sh` - Training launcher with CLI args
- `scripts/aws/run_evaluation.sh` - Evaluation launcher (standard + ablation)
- `docs/AWS_GUIDE.md` - Complete setup and usage guide

**Recommended instances:**
| Instance | GPU | VRAM | Cost/hr | Use Case |
|----------|-----|------|---------|----------|
| g5.xlarge | A10G | 24GB | ~$1.00 | Development |
| g5.2xlarge | A10G | 24GB | ~$1.20 | Full training |
| p3.2xlarge | V100 | 16GB | ~$3.00 | Faster training |

**Quick start:**
```bash
# On EC2 with Deep Learning AMI
git clone <repo> && cd BrainLLM
bash scripts/aws/setup.sh
source ~/activate.sh
./scripts/aws/run_training.sh --epochs 3
```

**Cost estimates:**
- WikiText-2 full (3 epochs): ~$6-9 on g5.xlarge
- Ablation study (15 variants × 3 seeds): ~$50 on g5.xlarge
- Spot instances: 60-70% savings

### Pending / next steps
- [ ] Authenticate with HuggingFace (`huggingface-cli login`) for Gemma access
- [ ] Add `--max-samples` flag to train.py for quick iteration
- [ ] Run initial WikiText-2 training (quick test with 1K samples)
- [ ] Execute ablation study
- [ ] Scale to PG-19/NarrativeQA (cloud GPU)
- [ ] Generate paper figures
- [ ] Write up results
