# CoRe-DF Implementation Plan

This plan operationalizes the requirements of the PRD v3.0, incorporating key architectural decisions resolved during our design review. It outlines the step-by-step execution path to validate the core hypothesis: **A deepfake detector's forensic evidence is destroyed differently depending on the *order* of lossy operations.**

---

## Phase 0: Repair Blocking Issues (1-2 days)
Before writing any new architectural code, we must stabilize the repository. This phase acts as a strict hard-gate.

1.  **CI Fix:** Update `.github/workflows/ci.yml` to replace the broken `libgl1-mesa-glx` with `libgl1`.
2.  **Import Fix:** Fix the broken fallback import in `src/evaluation/evaluate.py` (Line ~131) from `from datasets.dataset` to `from src.datasets.dataset`.
3.  **Artifact Cleanup:** Remove the stray `dataset.patch` from the repository root.
4.  **Dependencies:** Add `streamlit`, `plotly`, and `streamlit-image-comparison` to `requirements.txt`.
5.  **Persist Predictions:** Update `evaluate.py` to write per-sample predictions (video_id, group_id, tier, label, prob, abstain flag, checkpoint) to a new `deepfake_robustness/outputs/predictions/` directory.
6.  **Fail-Closed Demo:** Update `streamlit_app.py` to raise a visible UI error on checkpoint load failure, rather than silently falling back to randomly initialized weights.
7.  **Dynamic Thresholding:** In `streamlit_app.py`, use the dynamically calibrated threshold returned by `load_model_from_checkpoint` instead of a hardcoded `0.5`.
8.  **Cosmetic Dropdown:** Wire the model variant dropdown in `streamlit_app.py` to correctly select and load the appropriate checkpoints.
9.  **Label Correction:** Rename "Degradation Score" to "Confidence Delta" in the Streamlit UI to accurately reflect its current computation.

---

## Phase 1: Go/No-Go Measurement (3-5 days)
We will conduct a low-cost, high-signal test to confirm H1 (the residual non-commutativity hypothesis) before investing in the full architecture.

1.  **Data Generation:** Create a small paired dataset of **~5,000 FF++ frames**. For a given pair of operators (e.g., $o_1$ = JPEG, $o_2$ = Downscale-Upscale), generate $(o_1, o_2)$ and $(o_2, o_1)$ pipelines.
2.  **Iterative Severity Matching:** Implement an **Iterative SSIM matching** loop. Apply pipeline A with fixed parameters, then tune pipeline B's parameters (e.g., JPEG quality) until its absolute difference in SSIM to the original image compared to pipeline A is **$\le 0.01$**. This guarantees the probe detects order, not severity.
3.  **Feature Extraction:** Run these pairs through the existing, frozen `best_model.pt` feature extractor to obtain SRM and RGB embeddings.
4.  **Binary Probes:** Train a separate **Binary Logistic Regression** probe for each operator pair to classify which ordering produced the embedding.
5.  **Go/No-Go Gate:** If the probe achieves an AUC significantly > 0.5 (bootstrap CI excludes 0.5) on resample↔compress pairs, we proceed. If not, we abort and pivot to the v1/v2 framework.

---

## Phase 2: Compositional Data Engineering (1-2 weeks)
Develop the data pipeline to support compositional generalization testing.

1.  **Composable Transforms:** Refactor `src/degradations/transforms.py` into modular, deterministic, composable transform objects with explicit ordering control.
2.  **Dataset Splits:** Implement the compositional split generator using **6 structural/frequency operators** (JPEG, Downscale, Motion Blur, Gaussian Blur, Gaussian Noise, Sharpen - *dropping Color Jitter*):
    *   **Train:** Clean, all 6 single operators, all 30 ordered pairs (size 2).
    *   **Test:** Held-out combinations of size $\ge 3$, plus a specific *order-isolation* test set differing only in the sequence of two internal operations.
3.  **Severity Matching:** Port the Iterative SSIM matching logic (SSIM diff $\le 0.01$) from Phase 1 into the main data generation pipeline to ensure fair severity balancing across the order-isolation test set.
4.  **Configuration:** Update `config.py` to define explicit pipeline definitions (ordered lists of operators).

---

## Phase 3: Architecture Implementation (2-3 weeks)
Build the CoRe-DF modules inside `src/models/model.py`.

1.  **Modular Expert Heads ($E_1 \dots E_6$):** Add lightweight **MobileNetV3 blocks** followed by a global average pool and a linear layer (<50K parameters each) that process the shared RGB+SRM backbone features. Each head outputs an operator-specific feature vector $f_k$ and a validity/reliability score $v_k$. Add this as `branch_mode="modular"`.
2.  **Order Inference Module:** Implement a **Tiny Transformer Encoder (2 layers, 4 attention heads, d_model=64)**. This module will treat the $K$ expert feature vectors as an unordered set, using self-attention to capture interactions, and project the output into a pairwise relative-order logit matrix.
3.  **Order-Conditioned Fusion:** Build `branch_mode="modular_order"`, fusing the expert heads using a **Cross-Attention mechanism** where the order embeddings/logits guide the attention weights over the expert features.
4.  **Loss Function:** Implement the composite loss with fixed weights:
    *   Classification Loss: **1.0** (BCE/Focal)
    *   Order-prediction Loss: **0.5** (Cross-entropy on relative-order logits)
    *   Branch-reliability Loss: **0.5** (Self-supervised by expert correctness)
    *   Trajectory-consistency Loss: **0.1** (retained from v2)

---

## Phase 4: Training and Decisive Ablations (2-4 weeks)
Run the core experiments to validate the architecture.

1.  **Ablation Runs:** Train all 5 variants identified in the PRD (Monolithic, Modular-blind, Modular-order, Modular-order+Trajectory, Full CoRe-DF) across $\ge 3$ seeds.
2.  **Metrics:** Compute AUC, F1, Compositional Generalization Gap (CGG), and Order-Sensitivity Score (OSS).
3.  **Hypothesis Testing:** Run FDR-corrected paired bootstrap significance tests specifically on the *Modular-blind vs. Modular-order* outputs to isolate the value of order-conditioning.
4.  **Cross-Dataset Eval:** Test zero-shot transfer on platform-emulated tiers, Celeb-DF, DFDC, DeeperForensics, and Deepfake-Eval-2024.

---

## Phase 5: Packaging and Release (3-5 days)
1.  Generate final Model and Dataset cards including composition/order metadata.
2.  Compile the final report detailing which hypotheses (H1-H5) were supported or falsified.
3.  Tag the final release and merge the experimental branch into `main`.
