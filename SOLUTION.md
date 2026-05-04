# SMILES 2026 Summer School - Task Report

## Contents

1. How to run
2. Task characteristics
3. Method description
4. Results


### 1. How to run

#### Google Colab

Open the terminal in Colab and run:

```python
git clone https://github.com/MichaelToschenko/Hallucination-detection-task.git
cd Hallucination-detection-task
pip install -r requirements.txt
python solution.py
```

#### Local Setup

```bash
git clone https://github.com/MichaelToschenko/Hallucination-detection-task.git
cd Hallucination-detection-task

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```


### 2. Task characteristics

After studying the problem statement and running roughly **500** ablations across several iterations (the final iteration is summarised in `ablation_summary.csv`), I arrived at the following conclusions that shaped the final solution:

1. The small labelled set (689 examples) caps how complex the probe can be without overfitting. The final probe is a 2-hidden-layer MLP with 256 units per layer.
2. Mid-depth layers carry the strongest signal. The final solution uses three mid layers of Qwen with indices `-14, -12, -10` (i.e. transformer layers 11, 13, 15 out of 24).
3. Hand-crafted geometric features (norms, inter-layer drift, last-vs-mean, topology) gave no test-time gain and were disabled entirely.
4. The best pooling is a **combination** of two: the vector of the last real token plus the mean over the last 16 real response tokens.
5. With so little data, single-split estimates were unstable, so I moved to a 10-fold stratified CV.
6. Classes are imbalanced (~30% hallucinations / ~70% truthful) → stratified split + `pos_weight` in BCE loss.


### 3. Method description

End-to-end pipeline (driven by `solution.py`):

1. **Tokenization and Qwen forward pass.** `prompt + response` are concatenated, tokenized (`max_length=512`, `batch_size=4`) and fed into Qwen. The output is a tensor of shape `(25, seq_len, 896)`: 1 embedding layer + 24 transformer layers.

2. **Aggregation (`aggregation.py`).** I take three mid layers `(-14, -12, -10)`. From each layer I extract two pooled vectors over the real (non-padding) tokens:
   - `last` - the vector at the last real token (the end of the response);
   - `tail_mean` - the mean over the last 16 real tokens.

   Concatenated, this produces a feature of size `3 × 2 × 896 = 5376`.

3. **Probe classifier (`probe.py`).** An ensemble of **5 MLPs** with seeds 42..46. Each network has the architecture:

   ```
   Linear(5376 → 256) → GELU → Dropout(0.3)
   Linear(256  → 256) → GELU → Dropout(0.3)
   Linear(256  → 1)
   ```

   Training setup: `StandardScaler` on the features, `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` to counter class imbalance, Adam (lr=1e-3, weight_decay=1e-4, batch=64), up to 300 epochs with **early stopping on val_loss** (patience=30) on an internal 15% validation split (separate per ensemble member). The five networks' probabilities are averaged - that average is `predict_proba`.

4. **Threshold calibration.** `fit_hyperparameters` searches the external per-fold validation split exhaustively (every unique predicted probability plus a 101-point grid on `[0, 1]`) for the threshold that maximises F1.

5. **10-fold stratified CV (`splitting.py`).** Per fold: `fit` on train → `fit_hyperparameters` on val → `predict` on test. Metrics are averaged across folds.

6. **Final submission.** After CV the ensemble is retrained on the union of train+val from all folds and applied to `data/test.csv` → `predictions.csv`.


### 4. Results

Averaged over 10 folds:

| Checkpoint | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline (always `0`) | 0.7010 | 0.000 | - |
| Probe (test, 10-fold avg) | **0.7533** | **0.8426** | **0.7625** |

Gain over baseline: **+5.2 accuracy**.
