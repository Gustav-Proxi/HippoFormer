# HippoFormer Project - Summary for Sanika

## What is this project?

This is **HippoFormer** - a novel AI architecture that makes language models (like ChatGPT) smarter by giving them a "hippocampus" - the part of the brain responsible for memory.

### The Big Idea

Human brains don't remember everything equally. The hippocampus decides what's important to remember and what to forget. During sleep, it "replays" important memories to consolidate them.

**HippoFormer does the same thing for AI:**
- It learns which words/tokens are important (like the hippocampus detecting significant events)
- It stores important information in a memory buffer
- It "replays" and consolidates memories, prioritizing what matters
- This helps the AI remember important context better, especially in long documents

### Why does this matter?

Current AI models struggle with long texts - they forget things mentioned earlier. HippoFormer is designed to solve this by mimicking how human memory actually works.

---

## How the Code Works

```
Input Text (e.g., a story or document)
         │
         ▼
┌─────────────────────────────────────────────┐
│  Base Language Model (Gemma-2B)             │
│  - Frozen (not trained, saves memory)       │
│  - Uses LoRA (small trainable adapters)     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  SALIENCE GATE (the "importance detector")  │
│  - Learns which tokens matter               │
│  - Scores each token 0-1 (unimportant→important) │
│  - Like the brain detecting "this is worth remembering" │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  MEMORY BUFFER (the "hippocampus")          │
│  - Stores important tokens with priorities  │
│  - Does "replay" - revisits memories        │
│  - Older/less important memories fade       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  OUTPUT FUSION                               │
│  - Combines current thinking + memories     │
│  - Produces final predictions               │
└─────────────────────────────────────────────┘
         │
         ▼
    Output (next word prediction)
```

### Key Files

| File | What it does |
|------|--------------|
| `hippoformer/model.py` | Main model that ties everything together |
| `hippoformer/salience/gate.py` | The "importance detector" |
| `hippoformer/memory/buffer.py` | The memory system with replay |
| `hippoformer/drift/calibrator.py` | Keeps embeddings stable over time |
| `hippoformer/train.py` | Training script |
| `hippoformer/config.py` | All settings in one place |
| `evaluation/` | Tools to measure how well it works |

---

## What Vaishak is Currently Doing

### Training on RunPod (Cloud GPU)

The model is being trained on **RunPod** - a cloud service that rents powerful GPUs.

**Current Setup:**
- GPU: NVIDIA RTX 4090 (24GB memory)
- Cost: ~$0.59/hour when running
- Dataset: WikiText-2 (Wikipedia text for language modeling)
- Base model: Google's Gemma-2B (2 billion parameters)

**Training Process:**
1. The model reads Wikipedia text
2. It tries to predict the next word
3. When it gets it wrong, it learns from mistakes
4. The salience gate learns what's important
5. The memory buffer learns how to consolidate
6. After many iterations, the model improves

**Why cloud GPU?**
- Training AI models needs powerful hardware
- A MacBook would take days/weeks
- Cloud GPU (4090) takes hours
- Pay only for what you use

### Training Settings

```
Sequence length: 256-512 tokens (how much text at once)
Batch size: 1-8 (how many examples at once)
Epochs: 3 (how many times to see all data)
Learning rate: 0.0001 (how fast to learn)
```

---

## The Science Behind It

### Inspiration from Neuroscience

The hippocampus in your brain:
1. **Detects important events** - "Sharp wave ripples" (SPW-Rs) mark significant moments
2. **Replays memories** - During sleep, it replays experiences to consolidate them
3. **Prioritizes** - More emotional/important memories get replayed more

HippoFormer translates this to AI:
1. **Salience Gate** = Detecting important tokens (like SPW-Rs)
2. **Priority Buffer** = Memory storage with importance scores
3. **Replay Consolidation** = Re-processing important memories with decay

### What Makes This Novel

| Existing Approaches | HippoFormer's Innovation |
|---------------------|--------------------------|
| Single "surprise" score | Dual pathway (local + global) salience |
| Static memory retrieval | Multi-round replay with decay |
| No importance weighting | Explicit weights [1.0 - 5.0] |
| No drift handling | Embedding drift calibration |

---

## Project Goals

### Research Paper

The end goal is to publish a research paper showing that:
1. HippoFormer improves language model performance
2. The hippocampus-inspired design actually helps
3. Each component (salience, memory, replay) contributes

### Experiments Needed

1. **Perplexity** - How well does it predict text? (lower = better)
2. **Ablation studies** - Remove each component, see what breaks
3. **Long-context** - Does it help with very long documents?
4. **Comparison** - Is it better than existing approaches?

---

## Quick Commands

```bash
# Install the project
pip install -e ".[all]"

# Run tests
pytest

# Train locally (slow, for testing)
python -m hippoformer.train

# Evaluate
python -m evaluation.runner --datasets wikitext-2 --max-samples 100
```

---

## Current Status (as of Feb 2025)

- [x] Core architecture implemented
- [x] Training pipeline working
- [x] Evaluation framework built
- [x] Cloud deployment scripts ready
- [ ] Full training run (in progress on RunPod)
- [ ] Ablation experiments
- [ ] Paper writing

---

## Questions?

The code is well-documented. Key places to look:
- `CLAUDE.md` - Technical overview for developers
- `dev-notes.md` - Running log of changes
- `docs/` - Additional documentation

This project combines AI/ML with neuroscience principles to build a smarter memory system for language models.
