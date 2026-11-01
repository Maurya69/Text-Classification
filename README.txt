README
======

This submission contains two Python scripts for the human-vs-machine text
classification task (label "A" = human, label "B" = machine). Both scripts
expect train.json, test.json, and sample_submission.csv to be present in the
same working directory (not included in this ZIP).

1. sol_simplified.py
   -----------------
   Main pipeline used to produce the final submission (submission_v2.csv).
   It builds a two-stage stacked ensemble:
     - Stage 1 (base models, all evaluated out-of-fold with 5-fold CV):
         * SVM  - TF-IDF (uni/bi/tri-gram) features + LinearSVC
         * LM   - simple n-gram language models (uni/bi/tri-gram) fit
                  separately per class; features are log-likelihood-ratio
                  statistics between the two class language models
         * Stats - handcrafted sequence statistics (length, entropy,
                  repetition rate, token-frequency percentiles, lag-
                  autocorrelation, histogram of token rarity, etc.)
         * CNN  - a 1D convolutional neural network (multi-kernel-width,
                  max+mean pooling) trained on the raw token sequences
     - Stage 2 (meta-model): the out-of-fold predictions/features from all
       stage-1 models are concatenated and fed into a HistGradientBoosting
       classifier (bagged across multiple seeds/folds) to produce the final
       probability that a document is machine-written.
   Running this script prints per-model and final stacked CV accuracy, then
   writes submission_v2.csv and stack_test_prob.npy.

2. Alternative.py
   ---------------
   Exploratory/comparison script containing four alternative single- or
   two-stage modeling approaches that were tried instead of (or alongside)
   the final stacked pipeline in sol_simplified.py:
     - Alternative 1: mean-pooled learned token-embedding classifier (MLP)
     - Alternative 2: bidirectional LSTM classifier over token embeddings
     - Alternative 3: the same stage-1 features as sol_simplified.py
                       (SVM + LM + stats + CNN), combined with a plain
                       logistic regression stage-2 model
     - Alternative 4: the same stage-1 features, combined with a small
                       MLP stage-2 model
   For each alternative it prints out-of-fold accuracy/precision/recall/F1.
   This script is for comparison/reference only; it does not generate a
   submission file.

Note: train.json, test.json, and sample_submission.csv are intentionally
excluded from this ZIP.
