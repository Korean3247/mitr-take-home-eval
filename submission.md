# Research Engineering Intern: Take-Home Evaluation

**Name:** [NAME]
**Email:** [EMAIL]
**GitHub Fork:** https://github.com/Korean3247/mitr-take-home-eval

---

## Part A: Critical Analysis (25 pts)

### Weakness 1: CLUB Estimator Divergence Masked by Clamping

**Description:**
The CLUBSample estimator's MI loss hits its lower clamp boundary (-50) by epoch 2 and remains there for the rest of training. This occurs because the CLUB variational network (the internal MLP that approximates the posterior) is trained with the same optimizer and learning rate (2e-5) as the main classifier, instead of a separate optimizer with a higher learning rate suited for the MI network's rapid adaptation needs.

Once clamped, CLUB outputs a constant value and provides no meaningful gradient signal; it functions as a fixed regularizer rather than an adaptive MI estimator. Despite this failure, CLUB results are reported in the comparison table as a valid experimental condition (accuracy: 0.6940, contradiction rate: 0.4260), which misrepresents a broken method as a functioning one. Any reviewer familiar with CLUB would immediately identify this as a methodological integrity issue.

**Code Reference:**
- `CLUBSample` class: `mi_loss = mi_loss.clamp(-50, 50)`
- `optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)` -- CLUB parameters tied to main optimizer with no separation

**Proposed Fix:**
```python
# Separate optimizer for CLUB's internal network
mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-3)

# In the training loop, update MI estimator first
mi_optimizer.zero_grad()
mi_update_loss = mi_estimator.learning_loss(z_i, z_j)
mi_update_loss.backward()
mi_optimizer.step()
```
Alternatively, explicitly report CLUB as "failed to converge" and exclude it from quantitative comparisons, noting the failure as a finding.

---

### Weakness 2: Single-Run Results with No Statistical Significance Testing

**Description:**
All reported results are from a single training run with no fixed random seed and no variance estimation. The headline result, InfoNCE achieving +0.20% accuracy improvement (0.6980 to 0.7000), corresponds to a difference of just 3 samples out of 1,500 in the BoolQ validation set. This margin is well within the variance introduced by random weight initialization, data shuffling, and dropout stochasticity.

Additionally, the experimental design is inconsistent across model scales: the DistilBERT notebook evaluates 4 MI strategies (CLUB, InfoNCE, Cosine, CKA), while the BERT/RoBERTa notebook evaluates only 2 (Cosine, CKA). This prevents any systematic conclusion about which MI estimator generalizes across model architectures and makes the scaling analysis incomplete.

Top-tier NLP venues (ACL, EMNLP, ICLR) require at minimum 3 runs with mean +/- standard deviation to support quantitative claims. Without this, no accuracy or contradiction rate difference can be considered statistically meaningful.

**Code Reference:**
- No `torch.manual_seed()` or `np.random.seed()` set anywhere in either notebook
- BERT/RoBERTa notebook: `MI_ESTIMATORS = {"cosine": CosineSimMI, "cka": CKAMI}` -- InfoNCE and CLUB absent

**Proposed Fix:**
```python
SEEDS = [42, 123, 2024]
all_results = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    result = run_experiment(config)
    all_results.append(result)

mean_acc = np.mean([r['accuracy'] for r in all_results])
std_acc  = np.std( [r['accuracy'] for r in all_results])
print(f"Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
```
Apply all 4 MI strategies uniformly across all backbone models to ensure a complete and comparable ablation.

---

## Part B: Workshop Selection (10 pts)

**Selected Workshop:**
Foundations of Reasoning in Language Models (FoRLM) @ NeurIPS 2025
(https://reasoning-workshop.github.io/)

**Justification:**
FoRLM focuses on advancing the foundational understanding of reasoning in language models, specifically how reasoning emerges, where it fails, and how it can be improved through theoretical analysis and rigorous empirical studies. MITR directly addresses this scope: it proposes a representation-level mechanism (MI regularization between transformer layers) to improve logical consistency in yes/no reasoning tasks. The workshop's emphasis on *why* models succeed or fail at reasoning, rather than just benchmark gains, aligns well with our finding that MITR's effectiveness is backbone-dependent (helping RoBERTa but hurting BERT), which raises foundational questions about how pretraining paradigms shape a model's amenability to representation-level interventions. Furthermore, FoRLM accepts both positive and negative results with honest analysis, fitting our mixed findings.

**Related Paper 1:**
Elena Voita, Rico Sennrich, and Ivan Titov. *"The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives."* EMNLP 2019.
Uses mutual information and the Information Bottleneck framework to study how token representations evolve across transformer layers under MLM vs. MT objectives. This is the diagnostic counterpart to MITR: Voita et al. *measure* inter-layer MI evolution, while MITR *regularizes* it during training to reduce redundancy.

**Related Paper 2:**
Tao Li, Vivek Gupta, Maitrey Mehta, and Vivek Srikumar. *"A Logic-Driven Framework for Consistency of Neural Models."* EMNLP-IJCNLP 2019 (pp. 3924-3935).
Proposes regularizing neural model *outputs* with explicit logic rules to enforce prediction consistency in NLI. MITR provides a complementary approach: rather than imposing output-level logic constraints, it targets *internal representation diversity* to indirectly improve logical consistency, offering a representation-theoretic alternative to rule-based regularization.

---

## Part C: Experimental Design (25 pts)

### Experiment 1 (highest priority): MI Regularization Strength (lambda) Sensitivity Across Backbone Pretraining Paradigms

**Motivation:**
The most striking finding in the existing results is that MITR has opposite effects on BERT and RoBERTa: RoBERTa-Cosine reduces contradiction rate by 3.20% while BERT-Cosine *increases* it by 3.00%, despite both showing accuracy improvements. The current implementation uses a fixed lambda=0.01 across all backbones with no ablation. This experiment tests whether the reversal is an artifact of a single poorly-chosen lambda, or a genuine architectural difference.

**Hypothesis:**
BERT (pretrained with MLM+NSP) is more sensitive to MI regularization strength than RoBERTa (MLM-only) due to its next-sentence prediction objective already encouraging inter-layer coherence. Specifically: at lambda=0.01, BERT over-regularizes and degrades logical consistency, but a smaller lambda (<=0.003) will recover the contradiction rate reduction seen in RoBERTa. There exists a lambda* < 0.01 for BERT such that contradiction rate improves (decreases), while RoBERTa achieves its best contradiction reduction at lambda >= 0.01.

**Experimental Setup:**
- Models: BERT-base, RoBERTa-base
- MI strategy: Cosine (best-performing, no learnable parameters, stable)
- Lambda sweep: {0.001, 0.003, 0.01, 0.03, 0.1}
- Dataset: BoolQ (same split as original: 8000 train / 1500 val / 500 contradiction pairs)
- Seeds: 3 runs per condition (seeds 42, 123, 2024) to compute mean +/- std
- Total conditions: 5 lambda values x 2 models x 3 seeds = 30 runs

**Expected Outcome:**
- BERT will show a U-shaped or monotonically improving contradiction rate as lambda decreases, with the optimal lambda* around 0.001-0.003
- RoBERTa will maintain or improve contradiction reduction across a wider lambda range
- This would confirm that backbone pretraining paradigm determines the optimal regularization strength, providing a practical guideline for MITR deployment

**Falsification Condition:**
If BERT's contradiction rate does not improve at any lambda in {0.001, 0.003}, the backbone-dependency is structural rather than a tuning artifact, and MITR should not be applied to NSP-pretrained models without architectural modification.

**GPU Resource Estimate:**
- Single run: ~15 min on T4 (BERT/RoBERTa, BoolQ, 5 epochs, batch 32)
- 30 total runs x 15 min = ~7.5 hours
- With Colab A100: ~3 hours (5x speedup)
- Practical path: run 1 seed first (~2.5 hrs on T4) to confirm trend, add seeds 2-3 if time allows

---

### Experiment 2: Layer-Selective MITR -- Applying Regularization Only to Upper Transformer Layers

**Motivation:**
The existing implementation applies MI regularization uniformly to ALL consecutive layer pairs: `for i in range(len(hs) - 1)`. However, interpretability research (Tenney et al., 2019; Jawahar et al., 2019) consistently shows that lower transformer layers encode surface/syntactic features while upper layers encode semantic and task-specific representations. Applying MITR to lower layers may disrupt useful syntactic structure without contributing to logical consistency. This experiment isolates the effect to upper layers, where representational redundancy is most likely to impair reasoning.

**Hypothesis:**
Applying MITR only to the upper half of transformer layers (layers n/2 to n-1) will achieve equal or greater contradiction rate reduction than full-layer MITR, while reducing accuracy degradation observed in BERT, because lower-layer MI regularization disrupts syntactic representations that support surface-level reading comprehension on BoolQ.

**Experimental Setup:**
- Models: RoBERTa-base (12 layers), BERT-base (12 layers)
- MI strategy: Cosine
- Lambda: 0.01 (original setting, for fair comparison)
- Conditions: Baseline / Full MITR (layers 0-10) / Upper MITR (layers 6-10) / Lower MITR (layers 0-5)
- Dataset: BoolQ (same split)
- Code change required: filter `diffs` list by layer index before MI computation
- Seeds: 3 per condition

**Expected Outcome:**
- Upper MITR will match or improve full MITR on contradiction rate for RoBERTa
- Upper MITR will outperform full MITR on BERT (less contradiction rate increase or even reduction)
- Lower MITR will show the worst contradiction rate, confirming that lower-layer regularization is harmful

**Falsification Condition:**
If full MITR and upper MITR show no statistically significant difference (within 1 std), layer position is irrelevant and the uniform approach is justified.

**GPU Resource Estimate:**
- 4 conditions x 2 models x 3 seeds = 24 runs x ~15 min = ~6 hours on T4
- With A100: ~2.5 hours
- Lower priority than Experiment 1; run after lambda sweep confirms backbone-dependency trend

---

## Part D: Implementation (30 pts)

**Notebook:** `mitr_lambda_sensitivity.ipynb` (committed to this repository)

**Experiment:** Lambda sensitivity sweep for MITR-Cosine on BERT-base and RoBERTa-base, evaluating accuracy and contradiction rate across lambda in {0.001, 0.003, 0.01, 0.03, 0.1} on BoolQ.

**Results Summary:**

| Model | Condition | Accuracy | Contradiction Rate |
|-------|-----------|----------|--------------------|
| BERT | Baseline | 0.7013 | **0.6859** |
| BERT | lambda=0.001 | 0.7100 | 0.6987 (+1.3%) |
| BERT | lambda=0.003 | 0.7087 | 0.7115 (+2.6%) |
| BERT | lambda=0.01 | **0.7120** | 0.7051 (+1.9%) |
| BERT | lambda=0.03 | 0.7113 | 0.6923 (+0.6%) |
| BERT | lambda=0.1 | 0.7073 | 0.7308 (+4.5%) |
| RoBERTa | Baseline | 0.7740 | **0.6474** |
| RoBERTa | lambda=0.001 | 0.7753 | 0.6987 (+5.1%) |
| RoBERTa | lambda=0.003 | 0.7767 | 0.6795 (+3.2%) |
| RoBERTa | lambda=0.01 | **0.7900** | 0.6795 (+3.2%) |
| RoBERTa | lambda=0.03 | 0.7820 | 0.6795 (+3.2%) |
| RoBERTa | lambda=0.1 | 0.7240 | 0.7885 (+14.1%) |

**Discussion (150 words):**

Our lambda sweep refutes the hypothesis that backbone-conditioned regularization strength can resolve MITR's contradiction rate degradation. Across all five lambda values tested on both BERT and RoBERTa, MITR-Cosine consistently worsens contradiction rate relative to baseline, even at the smallest lambda=0.001. This is a negative result with two important implications.

First, the accuracy-consistency gap is real: MITR reliably improves accuracy (BERT best: +1.07% at lambda=0.01; RoBERTa best: +1.60% at lambda=0.01) while simultaneously degrading logical consistency. This suggests accuracy and contradiction rate are decoupled, and MI regularization on layer differences does not address the representational features responsible for logical consistency.

Second, the original finding that RoBERTa-Cosine reduces contradiction rate did not replicate under controlled conditions with fixed seeds, lending empirical support to our Part A critique that single-run results without variance estimation are unreliable. The divergence between original and replicated results underscores the need for multi-seed evaluation in MITR research.

---

## Part E: Abstract (max 200 words)

Mutual Information Transformer Regularization (MITR) penalizes representational redundancy between consecutive transformer layers to improve logical consistency. Prior work demonstrated modest gains on BoolQ using a fixed regularization strength (lambda=0.01) but revealed a puzzling backbone-dependent reversal: MITR improved RoBERTa's contradiction rate while worsening BERT's. We hypothesized this reversal was a tuning artifact and conducted a systematic lambda sweep across {0.001, 0.003, 0.01, 0.03, 0.1} on both BERT-base and RoBERTa-base using the parameter-free Cosine MI estimator on BoolQ. Our results refute this hypothesis: MITR worsens contradiction rate at every lambda for both backbones, even while improving accuracy (BERT +1.07%, RoBERTa +1.60% at lambda=0.01). Notably, the original finding that RoBERTa-Cosine reduces contradiction rate did not replicate under controlled seeding, confirming that single-run evaluations are insufficient. These findings reveal a fundamental accuracy-consistency decoupling in MITR: MI regularization on layer differences improves task performance but does not address the representational features governing logical consistency. Future work should explore alternative MI targets or layer-selective application.
