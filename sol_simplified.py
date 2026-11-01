import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

#Functiom to load the code
def load(path):
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]



# Function to tokenize bi-grams and tri-grams
def ngram_keys(s, k):
    n = len(s) - k + 1
    keys = np.zeros(n, dtype=np.int64)
    for j in range(k):
        keys = keys * 18438 + s[j:j + n]
    return keys

# LANGUAGE MODELING CODE 
#function to build lookup table
def count_ngrams(docs, k):
    keys = np.concatenate([ngram_keys(s, k) for s in docs if len(s) >= k])
    grams, counts = np.unique(keys, return_counts=True)
    ctx, start = np.unique(grams // 18438, return_index=True)
    ctx_counts = np.add.reduceat(counts, start)
    return grams, counts, ctx, ctx_counts


def build_lm(docs):
    uni = np.zeros(18438)
    for s in docs:
        np.add.at(uni, s, 1)
    return {"uni": uni, "total": uni.sum(), "bi": count_ngrams(docs, 2), "tri": count_ngrams(docs, 3)}

#Function 
def lookup(sorted_keys, values, query):
    idx = np.searchsorted(sorted_keys, query)
    idx[idx == len(sorted_keys)] = len(sorted_keys) - 1
    return np.where(sorted_keys[idx] == query, values[idx], 0)

#Function to return code for conditional probability of n-gram
def cond_prob(table, gram_keys, ctx_keys):
    grams, counts, ctx, ctx_counts = table
    num = lookup(grams, counts, gram_keys)
    den = lookup(ctx, ctx_counts, ctx_keys)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)

#Function to get token probabolities
def token_logprobs(lm, s):
    p1 = (lm["uni"][s] + 1) / (lm["total"] + 18438)
    p2 = cond_prob(lm["bi"], ngram_keys(s, 2), s[:-1])
    p = p1.copy()

    if len(s) < 3:
        p[1:] = 0.4 * p1[1:] + 0.6 * p2
    else:
        p3 = cond_prob(lm["tri"], ngram_keys(s, 3), ngram_keys(s[:-1], 2))
        p[1] = 0.4 * p1[1] + 0.6 * p2[0]
        p[2:] = 0.25 * p1[2:] + 0.30 * p2[1:] + 0.50 * p3
    return np.log(p)

#Function to get language modelling features
def lm_features(lm_a, lm_b, docs):
    out = np.zeros((len(docs), 11))
    for i, s in enumerate(docs):
        if len(s) < 2:
            continue
        a = token_logprobs(lm_a, s)
        b = token_logprobs(lm_b, s)
        d = b - a                      
        half = len(d) // 2
        out[i] = [d.mean(), d.std(), np.median(d), (d > 0).mean(),
                  d[:half].mean(), d[half:].mean(), d.sum(),
                  a.mean(), b.mean(), a.std(), b.std()]
    return out

#Function to 5-fold Language model and test it
def lm_oof(train_docs, test_docs, y, folds):
    train_out = np.zeros((len(train_docs), 11))
    test_out = np.zeros((len(test_docs), 11))
    for f, (tr_idx, va_idx) in enumerate(folds, 1):
        lm_a = build_lm([train_docs[i] for i in tr_idx if y[i] == 0])
        lm_b = build_lm([train_docs[i] for i in tr_idx if y[i] == 1])
        train_out[va_idx] = lm_features(lm_a, lm_b, [train_docs[i] for i in va_idx])
        test_out += lm_features(lm_a, lm_b, test_docs) / len(folds)
        print(f"  lm fold {f}/{len(folds)}")
    print(f"LM sign(llr_sum) acc = {accuracy_score(y, train_out[:, 6] > 0):.4f}")
    return train_out, test_out


# Function to get document statistics

def stat_features(docs, freq_log):
    feats = []
    for s in docs:
        n = len(s)
        if n < 3:
            feats.append([0.0] * 9)
            continue

        uniq, counts = np.unique(s, return_counts=True)
        p = counts / n
        entropy = -(p * np.log(p)).sum()

        bigram_rep = 1 - len(np.unique(ngram_keys(s, 2))) / (n - 1)
        trigram_rep = 1 - len(np.unique(ngram_keys(s, 3))) / (n - 2)

        seen50 = np.mean([s[t] in s[max(0, t - 50):t] for t in range(1, n)])

        lf = freq_log[s]
        half = n // 2

        feats.append([n, len(uniq), len(uniq) / n, entropy, bigram_rep, trigram_rep,
                      seen50, lf.mean(), lf[half:].mean() - lf[:half].mean()])
    return np.array(feats, dtype=np.float64)


# Function to train 5-fold SVM model which uses TF-IDF features

def svm_oof(all_docs, y, folds):
    n_train = len(y)
    all_str = [" ".join(map(str, s)) for s in all_docs]
    settings = {
        "bi": dict(ngram_range=(1, 2), min_df=2),
        "tri": dict(ngram_range=(1, 3), min_df=3, max_features=1_500_000),
        "uni": dict(ngram_range=(1, 1), min_df=1),
    }
    train_out = np.zeros((n_train, len(settings)))
    test_out = np.zeros((len(all_docs) - n_train, len(settings)))

    for j, (name, kw) in enumerate(settings.items()):
        vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, lowercase=False,
                              dtype=np.float32, **kw)
        X = vec.fit_transform(all_str).tocsr()
        X_train, X_test = X[:n_train], X[n_train:]
        for tr_idx, va_idx in folds:
            svm = LinearSVC(C=1.2, max_iter=100000).fit(X_train[tr_idx], y[tr_idx])
            train_out[va_idx, j] = svm.decision_function(X_train[va_idx])
            test_out[:, j] += svm.decision_function(X_test) / len(folds)
        print(f"svm {name}: oof acc = {accuracy_score(y, train_out[:, j] > 0):.4f}")
    return train_out, test_out


# CODE FOR CNN

def make_batch(docs, idx, train):
    L = max(8, min(512, max(len(docs[i]) for i in idx)))
    x = np.full((len(idx), L), 18438, dtype=np.int64)
    for row, i in enumerate(idx):
        s = docs[i]
        if len(s) > L:
            start = np.random.randint(0, len(s) - L + 1) if train else 0
            s = s[start:start + L]
        x[row, :len(s)] = s
    return torch.from_numpy(x).to("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, emb_dim=128, channels=128):
        super().__init__()
        self.emb = nn.Embedding(18438 + 1, emb_dim, padding_idx=18438)
        self.drop = nn.Dropout(0.25)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = nn.Sequential(nn.Linear(2 * channels * 3 + 1, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        mask = (x != 18438).unsqueeze(1)
        h = self.drop(self.emb(x)).transpose(1, 2)
        feats = []
        for conv in self.convs:
            z = torch.relu(conv(h))
            feats.append(z.masked_fill(~mask, -1e4).max(2).values)
            feats.append((z * mask).sum(2) / mask.sum(2).clamp(min=1))      # mean pool
        length = torch.log1p(mask.sum(2).float()) / 7.0
        return self.head(torch.cat(feats + [length], 1)).squeeze(1)


@torch.no_grad()
def cnn_predict(model, docs, idx, bs=128):
    model.eval()
    idx = np.asarray(idx)
    order = np.argsort([len(docs[i]) for i in idx])
    probs = []
    for b in range(0, len(order), bs):
        batch = idx[order[b:b + bs]]
        probs.append(torch.sigmoid(model(make_batch(docs, batch, train=False))).cpu().numpy())
    out = np.empty(len(idx))
    out[order] = np.concatenate(probs)
    return out


def cnn_oof(train_docs, test_docs, y, folds, epochs, bs):
    train_out = np.zeros(len(train_docs))
    test_out = np.zeros(len(test_docs))
    test_idx = np.arange(len(test_docs))

    for f, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(f)
        np.random.seed(f)
        model = CNN().to("cuda" if torch.cuda.is_available() else "cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=epochs * (len(tr_idx) // bs + 1))
        loss_fn = nn.BCEWithLogitsLoss()

        best_acc, best_val, best_test = -1, None, None
        for ep in range(epochs):
            model.train()
            perm = np.random.permutation(tr_idx)
            for b in range(0, len(perm), bs):
                idx = perm[b:b + bs]
                target = torch.tensor(y[idx], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
                opt.zero_grad()
                loss_fn(model(make_batch(train_docs, idx, train=True)), target).backward()
                opt.step()
                sched.step()

            val_pred = cnn_predict(model, train_docs, va_idx)
            acc = accuracy_score(y[va_idx], val_pred > 0.5)
            print(f"cnn fold {f} epoch {ep}: val acc {acc:.4f}")
            if acc > best_acc:
                best_acc, best_val = acc, val_pred
                best_test = cnn_predict(model, test_docs, test_idx)

        train_out[va_idx] = best_val
        test_out += best_test / len(folds)
        print(f"cnn fold {f} best: {best_acc:.4f}")

    print(f"CNN oof acc = {accuracy_score(y, train_out > 0.5):.4f}")
    return train_out, test_out


# stage II

def stack(Z_train, y, Z_test, seeds=(0, 1, 2)):
    oof = np.zeros(len(y))
    test_prob = np.zeros(len(Z_test))
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + 100 + seed)
        for tr_idx, va_idx in skf.split(Z_train, y):
            gb = HistGradientBoostingClassifier(
                max_iter=600, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=30, random_state=42 + seed)
            gb.fit(Z_train[tr_idx], y[tr_idx])
            oof[va_idx] += gb.predict_proba(Z_train[va_idx])[:, 1] / len(seeds)
            test_prob += gb.predict_proba(Z_test)[:, 1] / (5 * len(seeds))
    return oof, test_prob


def main():
    train, test = load("train.json"), load("test.json")
    y = np.array([1 if d["label"] == "B" else 0 for d in train])  # B = machine
    train_docs = [np.asarray(d["text"], dtype=np.int64) for d in train]
    test_docs = [np.asarray(d["text"], dtype=np.int64) for d in test]
    all_docs = train_docs + test_docs
    n_train = len(train)
    print(f"train={n_train} test={len(test)} machine share={y.mean():.3f}")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_train), y))

    # token frequencies over train+test (no labels used)
    freq = np.zeros(18438)
    for s in all_docs:
        np.add.at(freq, s, 1)
    stats = np.nan_to_num(stat_features(all_docs, np.log1p(freq)))

    svm_train, svm_test = svm_oof(all_docs, y, folds)
    lm_train, lm_test = lm_oof(train_docs, test_docs, y, folds)
    train_parts = [svm_train, lm_train, stats[:n_train]]
    test_parts = [svm_test, lm_test, stats[n_train:]]

    nn_train, nn_test = cnn_oof(train_docs, test_docs, y, folds, 8, 64)
    train_parts.append(nn_train[:, None])
    test_parts.append(nn_test[:, None])

    Z_train, Z_test = np.hstack(train_parts), np.hstack(test_parts)
    print("stacking matrix:", Z_train.shape)

    oof, test_prob = stack(Z_train, y, Z_test)
    print(f"stacked CV acc = {accuracy_score(y, oof > 0.5):.4f}")
    print(classification_report(y, oof > 0.5, target_names=["A human", "B machine"], digits=4))

    pred = np.where(test_prob > 0.5, "B", "A")
    ids = [d["id"] for d in test]
    sub = pd.read_csv("sample_submission.csv")
    assert list(sub["id"]) == ids, "test ids don't match sample_submission order"
    sub["label"] = pred
    sub.to_csv("submission_v2.csv", index=False)
    np.save("stack_test_prob.npy", test_prob)
    print(f"wrote submission_v2.csv (share B = {(pred == 'B').mean():.3f})")

main()
