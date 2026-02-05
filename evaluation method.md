# Evaluation Methodology: Neural-LLM Systems

This document outlines the dual-layer evaluation framework required to validate the accuracy of neural decoding and the quality of the LLM's generative response.

## 1. Tier 1: Neural Decoding Accuracy (Input Eval)
Before evaluating the LLM, we must ensure the "BrainGate" layer is providing accurate tokens.

* **Confusion Matrix Analysis**: Generate a confusion matrix for trial block decoding. [cite_start]Accuracy is validated if real data occupies the space immediately next to the diagonal[cite: 56, 132].
* [cite_start]**Null Model Testing**: Compare real data decoding against a **Poisson process model** (stochastic fluctuation) and **trial-shuffled data**[cite: 49, 50, 57].
* [cite_start]**Statistical Significance**: Decoding accuracy must be significantly higher than chance ($P < 10^{-10}$) to confirm structured variation[cite: 57].
* [cite_start]**Within-Event Coherence**: Verify that different time bins within a single SPW-R event coherently decode to the same trial block[cite: 244].

## 2. Tier 2: LLM Generative Quality (Output Eval)
Evaluate how effectively the LLM uses the neural signatures to generate contextually relevant text.

* [cite_start]**Contextual Relevancy**: Measure the correlation between the "Maze Replay" (active intent) and the LLM's generated output[cite: 303, 305].
* [cite_start]**Consolidation Metric**: Use the distribution of trial block identity during "post-experience sleep" (offline processing) as a predictor for model stabilization[cite: 252, 305].
* [cite_start]**Faithfulness**: Ensure the LLM output is grounded in the "tagged" neural patterns selected by the awake SPW-Rs[cite: 14, 327].

## 3. Advanced Diagnostic Metrics
| Metric | Purpose | Source |
| :--- | :--- | :--- |
| **Mean Error (Trial Blocks)** | Quantify the distance between predicted and actual neural states. [cite_start]| [cite: 56, 133] |
| **Pearson Correlation ($R$)** | Measure the relationship between awake tagging and sleep consolidation. [cite_start]| [cite: 303, 433] |
| **Trajectory Length** | Validate if a decoded neural event follows a significant path along the manifold. [cite_start]| [cite: 218, 301] |

## 4. Validation Workflow
1.  [cite_start]**Run "Leave-One-Trial-Out" validation**: The model should only be able to decode to the next closest trial block if the state space evolved in a structured way[cite: 54, 55].
2.  [cite_start]**Downsampling Stress Test**: Gradually reduce the number of neurons in the eval set to identify the "failure point" of the specific SLM being used[cite: 236].