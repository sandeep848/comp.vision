# Product Requirements Document (PRD)

## Robustness Benchmarking of Lightweight Deepfake Detectors Under Social-Media Compression, Resizing, and Content Degradation

### 1. Document Purpose

This Product Requirements Document defines the scope, objectives, methodology, technical requirements, evaluation framework, risks, success criteria, and deliverables for a research project on lightweight deepfake detection under realistic media degradation.

The project focuses on one practical deployment problem:

Deepfake detectors are often evaluated on clean benchmark data, while real-world content is commonly compressed, resized, downsampled, re-encoded, or otherwise transformed before analysis.

The purpose of this work is to quantify how much such transformations affect lightweight deepfake detectors and determine whether robustness-aware training can reduce the resulting performance loss.

The project is designed as a controlled robustness benchmarking and mitigation study rather than as a claim of major architectural novelty.

---

# 2. Project Title

Robustness Benchmarking of Lightweight Deepfake Detectors Under Social-Media Compression and Resizing

Alternative research title:

Evaluating Lightweight Deepfake Detectors Under Realistic Lossy Media Transformations

---

# 3. Chosen Domain

The selected application domain is:

Deepfake detection for social-media and messaging-platform content moderation.

Modern content moderation systems may receive media that has already undergone one or more processing operations before detection takes place.

Examples include:

* JPEG compression
* image resizing
* downsampling
* thumbnail generation
* transcoding
* multiple encoding passes
* resolution reduction
* blur
* noise
* recompression
* crop-and-rescale operations

These transformations are important because deepfake detection models often rely on subtle forensic artifacts that can be weakened or removed during processing.

The project therefore studies whether lightweight detectors remain useful once media passes through conditions closer to actual deployment.

---

# 4. Background

Deepfake generation has become increasingly realistic due to advances in generative adversarial networks, autoencoders, diffusion models, face-swapping pipelines, and neural rendering methods.

As generated and manipulated visual content becomes more convincing, automated detection becomes increasingly important for applications such as:

* social-media moderation
* misinformation detection
* identity verification
* digital forensics
* fraud prevention
* media authenticity analysis
* journalism and fact checking
* platform trust and safety systems

A large number of deepfake detection approaches achieve strong results on controlled datasets.

However, benchmark performance may not represent real deployment conditions.

Images uploaded to social-media platforms often do not remain in their original form.

A typical processing pipeline may look like:

Original Image
↓
Upload
↓
Resize
↓
Compression
↓
Re-encoding
↓
Thumbnail or transformed copy
↓
Deepfake detection system

Each transformation can modify image statistics and suppress manipulation-related artifacts.

A detector trained only on pristine benchmark data may therefore suffer from distribution shift when evaluated on processed media.

---

# 5. Problem Statement

Lightweight convolutional neural networks are attractive for practical deepfake detection because they offer lower computational cost, smaller memory requirements, faster inference, and easier deployment than very large architectures.

However, lightweight deepfake detectors are frequently evaluated using clean or relatively controlled benchmark images.

This creates a potential deployment gap.

Many deepfake detectors rely on weak visual signals such as:

* texture inconsistencies
* blending artifacts
* local frequency anomalies
* edge irregularities
* color inconsistencies
* resampling traces
* high-frequency noise patterns

Compression and resizing may weaken these signals.

For example, heavy JPEG compression can remove high-frequency components and introduce block artifacts.

Resizing can smooth local inconsistencies and alter interpolation patterns.

Downsampling may remove small manipulation cues entirely.

When transformations are combined, the information available to the detector can change substantially.

As a result, a detector with high performance on clean data may perform significantly worse after realistic media degradation.

The central problem is therefore:

> Clean benchmark performance may overestimate the real-world reliability of lightweight deepfake detectors when media undergoes compression, resizing, and other common platform transformations.

---

# 6. Motivation

A detector intended for practical content moderation must not only perform accurately on high-quality benchmark data.

It must remain reasonably reliable when input quality changes.

The project is motivated by four main concerns.

### 6.1 Deployment Realism

Online media rarely reaches moderation systems in exactly the same format in which it was created.

The model should therefore be tested under realistic post-processing conditions.

### 6.2 Lightweight Deployment

Large models may provide strong performance but are expensive to deploy at scale.

A lightweight detector can potentially support:

* faster inference
* lower cloud cost
* edge deployment
* mobile deployment
* real-time moderation
* high-throughput screening

Therefore, studying the robustness of lightweight architectures is practically relevant.

### 6.3 Evaluation Gap

A model reporting strong clean-data AUC may still be unsuitable for deployment if its AUC collapses after compression or resizing.

Robustness must therefore be evaluated explicitly.

### 6.4 Mitigation

If robustness-aware training can significantly reduce performance degradation without increasing model size, it may provide a practical method for improving deployment reliability.

---

# 7. Main Research Question

How much do compression and resizing reduce the performance of lightweight deepfake detectors, and can robustness-aware training reduce this performance loss?

---

# 8. Secondary Research Questions

The project will investigate several additional questions.

### RQ1

How much does detector performance decrease between clean and degraded test data?

### RQ2

Which degradation type causes the largest loss in performance?

### RQ3

How does degradation severity affect model performance?

### RQ4

Does robustness-aware training improve degraded-image performance compared with clean-only training?

### RQ5

Does robustness-aware training reduce performance on clean images?

### RQ6

Is there a measurable trade-off between clean-data accuracy and degraded-data robustness?

### RQ7

Does robustness-aware training improve cross-dataset generalization?

### RQ8

How stable are prediction confidence scores under progressively stronger degradation?

### RQ9

Are the models well calibrated under degraded conditions?

### RQ10

Can robustness be improved while preserving lightweight computational requirements?

### RQ11

Do some classes of degradations affect real and fake samples differently?

### RQ12

Are robustness improvements consistent across several degradation types, or specific only to transformations included during training?

---

# 9. Research Hypotheses

The project will test the following hypotheses.

### H1

Compression and resizing will reduce the performance of a lightweight deepfake detector compared with clean-data evaluation.

### H2

Performance loss will increase as degradation severity increases.

### H3

Strong compression and aggressive resizing will produce larger performance losses than weak degradation.

### H4

A robustness-aware model trained using simulated degradations will outperform a clean-trained baseline on degraded test data.

### H5

The robustness-aware model may show a small reduction in clean-data performance compared with the clean-only model.

### H6

The robustness-aware model will show a smaller relative performance drop from clean to degraded conditions.

### H7

Robustness-aware training may improve performance on an unseen external dataset, although cross-dataset generalization is not guaranteed.

### H8

Confidence calibration will deteriorate under degradation unless explicitly addressed during training.

---

# 10. Research Contribution

The main contribution should not be presented primarily as a novel architecture.

The strongest contribution is:

> A systematic robustness benchmarking and mitigation study for lightweight deepfake detectors under realistic lossy transformations.

The project contributes through:

* controlled comparison of clean and robustness-aware training
* degradation-specific evaluation
* severity-based robustness analysis
* external-dataset evaluation
* calibration analysis
* model-efficiency analysis
* interactive demonstration of degradation effects
* reproducible experimental configurations

The lightweight detector acts as the controlled experimental model.

---

# 11. Project Goals

The project has the following primary goals.

### Goal 1

Build a reliable baseline lightweight deepfake detector.

### Goal 2

Create a reproducible degradation benchmark.

### Goal 3

Quantify performance loss under compression and resizing.

### Goal 4

Train a robustness-aware version of the same model.

### Goal 5

Compare the two models under identical experimental conditions.

### Goal 6

Measure robustness-performance trade-offs.

### Goal 7

Evaluate generalization on an external dataset.

### Goal 8

Measure model efficiency.

### Goal 9

Analyze prediction confidence and calibration.

### Goal 10

Build an interactive demonstration for technical and non-technical users.

---

# 12. Non-Goals

The following are explicitly outside the primary scope.

### 12.1 Full Deepfake Detection Benchmarking Across Every Architecture

The project will not attempt to compare dozens of deepfake detection architectures.

### 12.2 State-of-the-Art Architectural Competition

The objective is not to outperform every existing detector.

### 12.3 Deepfake Generation

The project does not aim to generate new deepfakes.

### 12.4 Video Temporal Modeling as the Main Contribution

The primary experiments focus on image or frame-level detection.

Temporal video models may be explored only if time permits.

### 12.5 Production Deployment on a Real Social-Media Platform

The demonstration will simulate deployment-style transformations rather than integrate directly with a commercial platform.

### 12.6 Guaranteed Generalization to Every Manipulation Method

Cross-dataset testing will provide evidence of generalization but cannot prove universal robustness.

---

# 13. Target Users

Although this is primarily a research project, the resulting system has several potential user groups.

### Researcher

Needs:

* reproducible experiments
* detailed metrics
* degradation analysis
* comparable evaluation conditions
* experiment logs

### Content Moderator

Needs:

* understandable real/fake predictions
* confidence information
* awareness of uncertain cases

### Trust and Safety Engineer

Needs:

* reliable deployment information
* robustness-performance trade-offs
* inference cost
* failure cases

### Non-Technical User

Needs:

* simple visualization
* clear explanation of how image quality affects predictions
* direct comparison of models

---

# 14. Data Sources

## 14.1 Primary Dataset: FaceForensics++

FaceForensics++ will be the main dataset used for training, validation, and controlled evaluation.

It provides:

* authentic content
* manipulated content
* multiple manipulation methods
* compressed variants
* video-based data suitable for frame extraction
* standardized evaluation opportunities

Possible manipulation classes include:

* DeepFakes
* Face2Face
* FaceSwap
* NeuralTextures

The project may initially collapse these categories into binary labels:

Real
vs
Fake

---

## 14.2 External Dataset: Celeb-DF

Celeb-DF will be used as an external generalization benchmark.

The model will not be trained on Celeb-DF during the primary experiment.

Purpose:

* test domain shift
* test manipulation shift
* assess generalization
* compare robustness on unseen data

Where feasible, degraded versions of Celeb-DF samples will also be generated.

---

# 15. Dataset Unit of Analysis

The primary model will operate on face images or extracted video frames.

For each source video:

1. extract frames
2. detect faces
3. crop facial regions
4. apply consistent preprocessing
5. associate each crop with the original video ID
6. assign real/fake label
7. preserve metadata for split integrity

A fixed face detector and crop strategy must be used across all experiments.

---

# 16. Dataset Splitting

The target split is:

| Partition  | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Test       |        15% |

However, splitting must not be performed randomly at image level if that creates leakage.

The preferred unit of splitting is:

source video

or, where possible:

identity/source pair.

This is critical because adjacent frames from the same video are highly correlated.

A random frame-level split could inflate performance by allowing near-duplicate content to appear in both training and testing.

---

# 17. Data Leakage Controls

The system must include explicit leakage prevention.

Requirements:

* source video IDs must not overlap between train and test
* degraded variants of a test image must not appear in training
* frames from the same original sequence must remain in one split
* Celeb-DF must remain completely excluded from primary training
* preprocessing must not use test statistics
* hyperparameters must not be tuned on the final test set

A validation script should verify:

Train Videos ∩ Validation Videos = ∅

Train Videos ∩ Test Videos = ∅

Validation Videos ∩ Test Videos = ∅

---

# 18. Class Balance

The project should attempt to maintain reasonable balance between:

* real samples
* fake samples

If the dataset contains major class imbalance, one or more of the following may be used:

* balanced sampling
* class weighting
* stratified splits
* controlled subsampling

The exact strategy must be documented.

---

# 19. Preprocessing Pipeline

The preprocessing pipeline should remain identical between the baseline and robustness-aware models except for the degradation augmentation stage.

Example pipeline:

Input frame
↓
Face detection
↓
Face crop
↓
Optional margin expansion
↓
Resize to model input resolution
↓
Normalization
↓
Model input

For robustness-aware training:

Input frame
↓
Face detection
↓
Face crop
↓
Random degradation
↓
Resize
↓
Normalization
↓
Model input

---

# 20. Model Architecture

The project should use a lightweight convolutional backbone.

Recommended baseline:

EfficientNet-B0

Possible alternatives include:

* MobileNetV3
* EfficientNet-B1
* ConvNeXt-Tiny if resource constraints permit

The final architecture should satisfy the requirement of practical efficiency.

---

# 21. Standard Model

The Standard Model serves as the baseline.

Training characteristics:

* clean or minimally processed inputs
* no explicit robustness degradation curriculum
* same architecture as robustness-aware model
* same optimizer family
* same dataset split
* same input resolution
* same evaluation procedure

Its primary function is to establish:

clean performance

and

degradation sensitivity.

---

# 22. Robustness-Aware Model

The Robustness-Aware Model uses the same backbone but is trained using realistic image degradations.

Potential augmentations include:

* JPEG compression
* resizing
* downsampling
* blur
* noise
* repeated compression
* combined transformations

The model should not gain an architectural advantage over the baseline.

The comparison should isolate the effect of the training strategy.

---

# 23. Degradation Taxonomy

The benchmark will organize transformations into categories.

## 23.1 Clean

No additional degradation.

## 23.2 Compression

Example JPEG quality values:

* Q95
* Q90
* Q70
* Q50
* Q30

Possible interpretation:

| Level  |    JPEG Quality |
| ------ | --------------: |
| None   | 100 or original |
| Weak   |              90 |
| Medium |              70 |
| Strong |              50 |
| Severe |              30 |

---

# 24. Resizing

Possible resize scale factors:

* 1.00
* 0.75
* 0.50
* 0.25

Example:

Original size
↓
50% downsample
↓
Resize back to network input size

This helps isolate information loss caused by downsampling.

---

# 25. Combined Degradations

A more deployment-realistic benchmark will combine transformations.

Example:

Original
↓
Resize to 50%
↓
JPEG Q50
↓
Upscale to model input resolution

Combined configurations could include:

* Resize 75% + Q70
* Resize 50% + Q50
* Resize 25% + Q30

---

# 26. Optional Additional Degradations

If the core benchmark is complete, the following may be added.

### Gaussian Blur

Examples:

σ = 0.5
σ = 1.0
σ = 2.0

### Motion Blur

Short and medium motion kernels.

### Gaussian Noise

Low and moderate variance.

### Repeated Compression

Example:

Q90 → Q70 → Q50

This simulates multiple downloads and uploads.

### Re-encoding

Decode and encode again under a different configuration.

---

# 27. Degradation Curriculum

The robustness-aware model may use progressive augmentation difficulty.

Possible schedule:

### Early Training

* clean
* weak JPEG
* mild resizing

### Middle Training

* clean
* weak degradation
* medium compression
* medium resize

### Late Training

* weak
* medium
* strong
* combined degradations

A possible probability schedule:

Epochs 1–5:
70% clean, 30% degraded

Epochs 6–10:
40% clean, 60% degraded

Epochs 11+:
20% clean, 80% degraded

The exact schedule should be tested rather than assumed optimal.

---

# 28. Experimental Design

The minimum required comparison is:

| Training         | Testing  |
| ---------------- | -------- |
| Clean            | Clean    |
| Clean            | Degraded |
| Robustness-aware | Clean    |
| Robustness-aware | Degraded |

This results in the central controlled comparison.

A larger experiment matrix can then be constructed.

---

# 29. Benchmark Matrix

Example:

| Model    | Clean | JPEG 90 | JPEG 70 | JPEG 50 | JPEG 30 | Resize 75% | Resize 50% | Resize 25% | Combined |
| -------- | ----: | ------: | ------: | ------: | ------: | ---------: | ---------: | ---------: | -------: |
| Standard |     ✓ |       ✓ |       ✓ |       ✓ |       ✓ |          ✓ |          ✓ |          ✓ |        ✓ |
| Robust   |     ✓ |       ✓ |       ✓ |       ✓ |       ✓ |          ✓ |          ✓ |          ✓ |        ✓ |

For every cell, report:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* ECE where available

---

# 30. Primary Evaluation Metric

ROC-AUC should be used as one of the main evaluation metrics.

Reasons:

* threshold independent
* suitable for binary classification
* allows comparison across different operating points
* widely used in deepfake detection

AUC alone, however, should not be treated as sufficient.

---

# 31. Secondary Evaluation Metrics

The project should report:

### Accuracy

Overall correct classification rate.

### Precision

Measures false-positive sensitivity.

### Recall

Measures fake detection coverage.

### F1-Score

Balances precision and recall.

### Specificity

Useful for understanding false positives on real content.

### False Positive Rate

Important for moderation systems because excessive false positives can create operational problems.

### False Negative Rate

Important because missed deepfakes may remain unmoderated.

---

# 32. Threshold Selection

A fixed classification threshold should not automatically be assumed optimal.

The project should define one of the following:

* threshold = 0.5
* threshold chosen on validation data
* equal-error-rate threshold
* threshold selected for target recall

The threshold used for test evaluation must be chosen before accessing final test labels.

---

# 33. Robustness Metrics

The central research objective requires dedicated robustness measurements.

## 33.1 Absolute Performance Drop

For metric \(M\):

$$
\Delta M = M_{clean} - M_{degraded}
$$

Example:

Clean AUC = 0.82

JPEG Q30 AUC = 0.64

Absolute AUC drop = 0.18

---

# 34. Relative Performance Drop

$$
RelativeDrop = \frac{M_{clean} - M_{degraded}}{M_{clean}}
$$

This allows comparison across models with different clean baselines.

---

# 35. Robustness Retention

Another useful metric:

$$
Retention = \frac{M_{degraded}}{M_{clean}}
$$

Example:

Clean AUC = 0.80
Degraded AUC = 0.72

Retention = 90%

This can be easier to communicate in the interactive demo.

---

# 36. Mean Robustness Score

An optional aggregate metric can summarize degradation performance.

For example:

$$
MeanRobustAUC = \frac{1}{N}\sum_{i=1}^{N} AUC_i
$$

where each \(i\) represents a degradation condition.

This should supplement, not replace, condition-specific results.

---

# 37. Statistical Reliability

Single performance numbers should not be interpreted without uncertainty.

Where feasible, the project should report:

* bootstrap confidence intervals
* standard deviation across runs
* repeated-seed experiments
* significance testing between models

Example:

AUC = 0.78
95% CI = [0.76, 0.80]

For major model comparisons, paired bootstrapping may be used.

---

# 38. Multiple Random Seeds

At least 3 random seeds should be considered for the strongest claims if computational resources allow.

Example:

Seed 42
Seed 123
Seed 2026

Report:

Mean ± Standard Deviation

This reduces the chance of reporting an unusually favorable training run.

---

# 39. Calibration Evaluation

A moderation system should not only classify correctly.

Its confidence should also be meaningful.

The project should therefore evaluate:

### Expected Calibration Error

ECE measures the difference between model confidence and empirical correctness.

### Reliability Diagram

Confidence bins plotted against observed accuracy.

### Confidence Distribution

Compare confidence distributions for:

* clean real
* clean fake
* degraded real
* degraded fake

---

# 40. Confidence Degradation

The project should measure how confidence changes as degradation severity increases.

For example:

| JPEG Quality | Fake Confidence |
| ------------ | --------------: |
| 100          |            0.94 |
| 90           |            0.91 |
| 70           |            0.83 |
| 50           |            0.69 |
| 30           |            0.54 |

This can reveal instability before the final prediction actually changes.

---

# 41. Cross-Dataset Evaluation

Celeb-DF will be used as an unseen dataset.

Evaluation configurations should include:

* clean Celeb-DF
* JPEG-compressed Celeb-DF
* resized Celeb-DF
* combined degradation

The model should not be fine-tuned on Celeb-DF before this evaluation.

---

# 42. Interpretation of Cross-Dataset Results

Cross-dataset performance measures two different effects:

### Domain Generalization

Can the model handle different manipulation characteristics?

### Degradation Robustness

Can the model handle lower-quality media?

These effects should not be conflated.

For example, poor Celeb-DF performance may arise even on clean images because of domain shift.

---

# 43. Efficiency Requirements

Because the project focuses on lightweight detectors, efficiency must be measured.

Required measurements should include as many as practical:

* parameter count
* model file size
* FLOPs or MACs
* GPU inference latency
* CPU inference latency
* throughput
* VRAM usage
* RAM usage

---

# 44. Efficiency Success Principle

The robustness-aware model should ideally improve degraded-data performance without increasing inference-time complexity.

Because both models use the same architecture:

parameter count should remain effectively identical.

The primary additional cost should occur during training, not deployment.

This is a key practical advantage.

---

# 45. Error Analysis

The project should analyze failure cases instead of reporting aggregate metrics only.

For misclassified samples, investigate patterns such as:

* extreme blur
* low resolution
* side-profile faces
* occlusion
* unusual lighting
* heavy makeup
* compression artifacts
* very small faces
* multiple faces
* manipulated regions outside the face crop

---

# 46. Real-vs-Fake Error Breakdown

Results should separately report:

False Real:

fake image incorrectly classified as real

False Fake:

real image incorrectly classified as fake

This distinction is operationally important.

---

# 47. Degradation-Specific Failure Analysis

For every degradation class, record:

* number of errors
* false-positive rate
* false-negative rate
* confidence distribution
* AUC degradation
* misclassification examples

This enables a detailed understanding of failure modes.

---

# 48. Feature-Space Analysis

Optional exploratory analysis may examine whether degradation moves samples in embedding space.

Possible methods:

* PCA
* t-SNE
* UMAP

Compare:

clean real
clean fake
degraded real
degraded fake

This may provide qualitative insight into distribution shift.

It should be considered supplementary, not a core requirement.

---

# 49. Ablation Studies

If time permits, perform controlled ablations.

Examples:

### Compression Only

Train with JPEG compression but no resize augmentation.

### Resize Only

Train with resizing but no compression.

### Combined Training

Train with both.

### Curriculum vs Random Severity

Compare progressively stronger augmentation with randomly sampled severity.

### Clean Sample Retention

Compare different proportions of clean samples during robust training.

These experiments can identify which components produce the robustness improvement.

---

# 50. Functional Requirements

The research software must support:

* dataset loading
* face extraction
* deterministic dataset splits
* augmentation configuration
* standard training
* robustness-aware training
* checkpoint saving
* checkpoint loading
* batch inference
* metric calculation
* robustness evaluation
* calibration evaluation
* cross-dataset evaluation
* model-efficiency measurement
* result export
* visualization

---

# 51. Configuration Requirements

Experiment parameters should be stored in configuration files.

Example categories:

* dataset paths
* split seed
* image size
* batch size
* learning rate
* optimizer
* weight decay
* scheduler
* augmentation probabilities
* JPEG quality ranges
* resize ranges
* training epochs
* early stopping
* model backbone

This improves reproducibility.

---

# 52. Logging Requirements

Each experiment should log:

* experiment name
* timestamp
* git commit hash where possible
* dataset split
* configuration
* model
* seed
* epoch
* training loss
* validation loss
* validation metrics
* final test metrics
* checkpoint path

Possible tools:

* CSV
* JSON
* TensorBoard
* Weights & Biases if permitted

---

# 53. Model Checkpointing

The system should save:

* latest checkpoint
* best validation checkpoint
* optional best AUC checkpoint

The final test evaluation should use the checkpoint selected using validation data only.

---

# 54. Early Stopping

Early stopping should be based on a validation metric such as:

validation AUC

or

validation loss.

Test data must not influence early stopping.

---

# 55. Training Reproducibility

The project should control:

* Python random seed
* NumPy seed
* PyTorch seed
* CUDA seed
* deterministic options where practical

Full determinism may reduce speed and is not always guaranteed on GPUs, so the exact environment should also be recorded.

---

# 56. Environment Reproducibility

The repository should contain:

* environment.yml
  or
* requirements.txt

Recommended to record:

* Python version
* PyTorch version
* CUDA version
* GPU model
* operating system

---

# 57. Repository Structure

A recommended structure is:

```text
project/
│
├── configs/
│   ├── baseline.yaml
│   ├── robust.yaml
│   └── evaluation.yaml
│
├── data/
│
├── datasets/
│   ├── faceforensics.py
│   └── celebdf.py
│
├── degradations/
│   ├── jpeg.py
│   ├── resize.py
│   ├── blur.py
│   └── pipeline.py
│
├── models/
│   └── detector.py
│
├── training/
│   ├── train.py
│   └── losses.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   ├── calibration.py
│   └── robustness.py
│
├── tests/
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│
├── README.md
└── environment.yml
```

---

# 58. Interactive Demo

A web application should communicate the core research result visually.

Recommended framework:

Streamlit

Alternative:

Gradio

---

# 59. Demo User Flow

The expected interaction is:

Upload image
↓
Face detected
↓
Original preview shown
↓
Choose degradation settings
↓
Degraded preview generated
↓
Run both models
↓
Display predictions
↓
Display confidence comparison
↓
Increase distortion
↓
Plot confidence degradation curve

---

# 60. Demo Inputs

The application should support:

### Required

Image upload

Possible formats:

* JPG
* JPEG
* PNG

### Optional

Short video clip

Possible formats:

* MP4
* MOV

Video support should be secondary because it increases processing complexity.

---

# 61. Demo Degradation Controls

The application should provide sliders or selectors for:

### JPEG Quality

Example:

30–100

### Resize Scale

Example:

25%–100%

### Blur

Optional.

### Noise

Optional.

### Combined Mode

Apply several distortions together.

---

# 62. Demo Output

The application should display:

### Original Image

Unmodified input.

### Degraded Image

Input after selected transformations.

### Standard Model

Prediction:

Real or Fake

Confidence:

percentage or probability.

### Robustness-Aware Model

Prediction:

Real or Fake

Confidence.

---

# 63. Demo Comparison Table

Example:

| Model    | Clean Prediction | Clean Confidence | Degraded Prediction | Degraded Confidence | Confidence Drop |
| -------- | ---------------- | ---------------: | ------------------- | ------------------: | --------------: |
| Standard | Fake             |              94% | Real                |                 48% |             46% |
| Robust   | Fake             |              91% | Fake                |                 76% |             15% |

This directly demonstrates robustness.

---

# 64. Demo Graph

The application should generate a curve such as:

X-axis:

JPEG Quality

Y-axis:

Fake probability

Two lines:

* Standard Model
* Robust Model

This can visually show the rate at which confidence deteriorates.

---

# 65. Demo Explanation Panel

For non-technical users, include a short explanation such as:

“Compression can remove small visual artifacts that deepfake detectors use for classification. A robust model should maintain more stable predictions as image quality decreases.”

---

# 66. User Stories

### Researcher

As a researcher, I want to evaluate the same detector across multiple degradation levels so that I can quantify robustness.

### Developer

As a developer, I want configuration-driven experiments so that I can reproduce results without changing source code.

### Moderator

As a moderator, I want to see model confidence after compression so that I can understand whether the prediction is reliable.

### Non-Technical User

As a user, I want to adjust image quality and see how detection changes so that I can understand why robustness matters.

---

# 67. Acceptance Criteria for Dataset Pipeline

The dataset pipeline is complete when:

* FaceForensics++ loads successfully
* labels are correct
* source IDs are retained
* splits are reproducible
* leakage tests pass
* class statistics are generated
* face crops can be reproduced
* degraded variants can be generated deterministically

---

# 68. Acceptance Criteria for Baseline Model

The baseline is complete when:

* model trains without errors
* validation metrics are logged
* best checkpoint is saved
* test metrics are reproducible
* clean baseline metrics are documented
* model efficiency is measured

---

# 69. Acceptance Criteria for Robust Model

The robustness-aware model is complete when:

* same architecture is used
* degradation augmentations are applied during training
* training settings are recorded
* clean and degraded test sets are evaluated
* robustness metrics are generated
* comparison with baseline is possible

---

# 70. Acceptance Criteria for Evaluation

Evaluation is complete when the following are generated for every primary degradation:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* confusion matrix
* absolute performance drop
* relative performance drop
* degradation curve

Where feasible:

* ECE
* confidence interval

---

# 71. Acceptance Criteria for External Testing

External evaluation is complete when:

* Celeb-DF is loaded independently
* no Celeb-DF samples appear in training
* clean results are generated
* degraded results are generated
* standard and robust models are compared

---

# 72. Acceptance Criteria for Demo

The demo is complete when a user can:

1. upload an image
2. view the original
3. modify degradation
4. view the distorted version
5. run the standard model
6. run the robust model
7. view both predictions
8. view both confidence scores
9. view confidence change across distortion
10. compare the models visually

---

# 73. Performance Requirements

The application should aim for:

Image inference:

within a few seconds

Model loading:

once at application startup

User interface:

responsive enough for interactive experimentation

The exact latency target depends on deployment hardware.

---

# 74. Hardware Constraints

The training pipeline should be suitable for university GPU infrastructure.

The lightweight model should ideally train on a single GPU.

Preferred characteristics:

* batch size adjustable
* mixed precision support
* checkpoint resume support
* configurable worker count

---

# 75. Risks

## Risk 1: Data Leakage

Severity: Critical

Mitigation:

Use video-level splitting and automated leakage tests.

---

# 76. Risk 2: Overfitting

Severity: High

Potential causes:

* limited identities
* excessive frame similarity
* excessive training epochs
* strong augmentation memorization

Mitigation:

* validation monitoring
* early stopping
* source-level split
* regularization
* multiple seeds

---

# 77. Risk 3: Dataset Bias

FaceForensics++ may not fully represent real-world deepfakes.

Mitigation:

Use Celeb-DF as external evaluation and clearly state dataset limitations.

---

# 78. Risk 4: Overfitting to Degradation Types

A robustness-aware model may become good only at degradations seen during training.

Mitigation:

Evaluate both seen and unseen degradation conditions where feasible.

---

# 79. Risk 5: Clean Performance Loss

Aggressive degradation training may hurt clean accuracy.

Mitigation:

Retain clean samples during robust training and tune augmentation probability.

---

# 80. Risk 6: Misleading Confidence

A model may produce high confidence even when wrong.

Mitigation:

Include calibration analysis.

---

# 81. Risk 7: Excessive Experiment Count

The number of combinations can grow rapidly.

Mitigation:

Define a core benchmark and separate optional experiments.

Core benchmark:

* clean
* 3 compression levels
* 3 resize levels
* 3 combined levels

---

# 82. Risk 8: Compute Limitations

Large experiment grids can become expensive.

Mitigation:

* lightweight model
* fixed input resolution
* mixed precision
* limited seeds during development
* full repeated runs only for final experiments

---

# 83. Ethical Considerations

The project deals with manipulated facial content.

Important considerations include:

* dataset licensing
* privacy
* responsible handling of biometric content
* avoiding misuse
* clear communication of uncertainty

The demo should not claim that a prediction is definitive proof that an image is authentic or manipulated.

---

# 84. Demo Disclaimer

The application should include a statement such as:

“This system is a research prototype. Predictions should not be treated as definitive evidence that media is real or fake. Model reliability may vary across datasets, image qualities, and manipulation methods.”

---

# 85. Research Limitations

The final report should explicitly discuss:

* dataset limitations
* manipulation coverage
* lack of full social-media pipeline replication
* image/frame-level focus
* model architecture scope
* external-domain shift
* computational constraints
* calibration limitations
* generalization limitations

---

# 86. Minimum Viable Project

The MVP should include:

* FaceForensics++ preprocessing
* video-level dataset split
* EfficientNet-B0 baseline
* robust version of same model
* JPEG compression
* resizing
* combined degradation
* clean and degraded evaluation
* ROC-AUC, F1, precision, recall, accuracy
* robustness drop
* basic Streamlit demo

---

# 87. Extended Project Scope

If time permits:

* blur
* motion blur
* noise
* repeated compression
* calibration
* bootstrap confidence intervals
* cross-dataset Celeb-DF robustness
* ablation studies
* multiple training seeds
* model latency benchmarking
* video upload support

---

# 88. Development Phases

### Phase 1 — Data Preparation

Tasks:

* acquire datasets
* inspect dataset structure
* build metadata
* extract frames
* detect faces
* create train/validation/test split
* implement leakage tests

Deliverable:

validated dataset pipeline.

---

# 89. Phase 2 — Baseline Model

Tasks:

* implement lightweight detector
* train clean baseline
* optimize training
* record metrics
* save checkpoint

Deliverable:

standard clean-trained model.

---

# 90. Phase 3 — Degradation Pipeline

Tasks:

* JPEG degradation
* resize degradation
* combined degradation
* severity configuration
* deterministic testing

Deliverable:

controlled degradation module.

---

# 91. Phase 4 — Robust Training

Tasks:

* add degradation-aware augmentation
* implement curriculum if used
* train robust model
* tune augmentation strength

Deliverable:

robustness-aware model.

---

# 92. Phase 5 — Benchmarking

Tasks:

* run degradation matrix
* calculate metrics
* calculate robustness drop
* generate ROC curves
* generate robustness curves
* compare models

Deliverable:

primary benchmark results.

---

# 93. Phase 6 — External Evaluation

Tasks:

* load Celeb-DF
* run clean evaluation
* apply degradation
* compare model generalization

Deliverable:

cross-dataset results.

---

# 94. Phase 7 — Calibration and Statistical Analysis

Tasks:

* reliability diagram
* ECE
* confidence distributions
* bootstrap confidence intervals
* significance analysis where applicable

Deliverable:

reliability analysis.

---

# 95. Phase 8 — Demo

Tasks:

* build application
* load models
* upload image
* apply degradations interactively
* display predictions
* generate robustness curve

Deliverable:

interactive prototype.

---

# 96. Phase 9 — Final Analysis

Tasks:

* analyze strongest and weakest conditions
* compare standard vs robust
* identify clean/robustness trade-off
* document limitations
* prepare final plots and tables

Deliverable:

research report.

---

# 97. Recommended Result Tables

The final report should include a clean-vs-degraded table.

Example:

| Condition | Standard AUC | Robust AUC | Δ Standard | Δ Robust |
| --------- | -----------: | ---------: | ---------: | -------: |
| Clean     |         0.82 |       0.80 |       0.00 |     0.00 |
| JPEG 70   |         0.74 |       0.78 |      -0.08 |    -0.02 |
| JPEG 50   |         0.68 |       0.75 |      -0.14 |    -0.05 |
| JPEG 30   |         0.60 |       0.70 |      -0.22 |    -0.10 |

The values above are illustrative only.

---

# 98. Recommended Figures

The final report should ideally include:

* ROC curves on clean data
* ROC curves under degradation
* AUC vs JPEG quality
* AUC vs resize ratio
* confidence vs degradation
* reliability diagram
* confusion matrices
* model performance vs latency

---

# 99. Key Decision Rule

The robustness-aware approach should be considered successful if it meaningfully improves degraded-data performance while preserving acceptable:

* clean performance
* inference speed
* model size

A small clean-data performance reduction may be acceptable if robustness improves substantially.

---

# 100. Example Interpretation

A meaningful result could look like:

Standard model:

Clean AUC = 0.82
Strong compression AUC = 0.61

Robust model:

Clean AUC = 0.79
Strong compression AUC = 0.73

Interpretation:

The robust model loses 0.03 clean AUC but reduces degradation-induced loss from 0.21 to 0.06.

This would support the argument that robustness-aware training provides a better deployment trade-off.

---

# 101. Failure Scenario

The project remains scientifically useful even if robustness-aware training does not outperform the baseline.

Possible result:

* robustness improves for JPEG
* resize robustness does not improve
* clean performance decreases
* external generalization remains poor

This would suggest that simple degradation-aware training is insufficient for broad deployment robustness.

Negative results should therefore not be treated as project failure.

---

# 102. Final Expected Outcome

The project is expected to produce a controlled and reproducible evaluation of how lightweight deepfake detectors behave under realistic lossy transformations.

The final analysis should answer:

* How much does performance drop?
* Under which distortions does it drop most?
* How quickly does performance degrade?
* Does robust training help?
* How much clean performance is sacrificed?
* Does robustness transfer across datasets?
* Are model confidence scores trustworthy?
* Is the resulting detector still computationally lightweight?

---

# 103. Final Research Story

The strongest narrative for the project is:

Problem
↓
Deepfake detectors are often evaluated on clean benchmark content, while real deployment pipelines introduce compression and resizing.

Deployment Gap
↓
These transformations can remove or distort forensic signals used by deepfake detectors.

Research Question
↓
How robust is a lightweight detector to such transformations?

Controlled Method
↓
Same detector architecture

* standard clean training
  vs
* robustness-aware degradation training

Evaluation
↓
Clean data
× compression
× resizing
× combined degradation
× multiple severity levels
× FaceForensics++
× external Celeb-DF

Measurements
↓
AUC

* Accuracy
* Precision
* Recall
* F1
* robustness drop
* calibration
* confidence stability
* efficiency

Outcome
↓
Determine whether degradation-aware training provides a better balance between clean accuracy, robustness, and lightweight deployment suitability.

---

# 104. Final Deliverables

The completed project should provide:

1. A reproducible FaceForensics++ preprocessing pipeline.

2. A leakage-safe dataset split.

3. A standard lightweight deepfake detector.

4. A robustness-aware version of the same detector.

5. A configurable degradation engine.

6. Compression robustness evaluation.

7. Resize robustness evaluation.

8. Combined degradation evaluation.

9. Clean-vs-degraded comparison tables.

10. Robustness curves.

11. ROC curves.

12. Accuracy, precision, recall, F1, and AUC measurements.

13. Relative and absolute performance-drop analysis.

14. Calibration analysis where feasible.

15. Confidence-stability analysis.

16. Cross-dataset Celeb-DF evaluation.

17. Model-size and inference-efficiency measurements.

18. Ablation experiments where feasible.

19. Failure-case analysis.

20. Interactive Streamlit or Gradio demonstration.

21. Reproducible configuration files.

22. Experiment logs and checkpoints.

23. Documentation explaining setup and execution.

24. Final research report.

25. A clear conclusion about whether clean-data evaluation is sufficient for estimating the deployment robustness of lightweight deepfake detectors.

---

# 105. Final Project Statement

This project investigates whether lightweight deepfake detectors remain reliable after image transformations that commonly occur in real-world media pipelines.

Rather than optimizing only for clean benchmark accuracy, the project measures robustness as a first-class evaluation criterion.

By comparing the same lightweight architecture under clean and degradation-aware training, the study isolates the effect of robustness-focused training while controlling model capacity and inference complexity.

The intended contribution is therefore not simply a detector that produces a real-or-fake prediction. It is a reproducible framework for understanding when lightweight deepfake detectors fail, how severely they fail, and whether realistic degradation-aware training can make them more suitable for practical content moderation.
