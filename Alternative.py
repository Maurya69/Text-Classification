import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

vocab_size = 18438
pad_id = 18438
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# load data, works for both json list and jsonl
def load(path):
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

train = load("train.json")
y = np.array([1 if d["label"] == "B" else 0 for d in train])
docs = [np.asarray(d["text"], dtype=np.int64) for d in train]
print("train size:", len(docs))
print("machine share:", y.mean())

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

# just prints acc/precision/f1 for a model's oof probs
def print_results(name, y_true, prob):
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, average=None, labels=[0, 1], zero_division=0)
    f1 = f1_score(y_true, pred, average=None, labels=[0, 1], zero_division=0)
    recall = recall_score(y_true, pred, average=None, labels=[0, 1], zero_division=0)
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"accuracy: {acc:.4f}")
    print(f"a human  | precision: {precision[0]:.4f} | f1: {f1[0]:.4f} | recall: {recall[0]:.4f}")
    print(f"b machine | precision: {precision[1]:.4f} | f1: {f1[1]:.4f} | recall: {recall[1]:.4f}")

# pads/crops docs to same len, random crop if training
def make_batch(docs, idx, train_mode):
    length = max(8, min(512, max(len(docs[i]) for i in idx)))
    x = np.full((len(idx), length), pad_id, dtype=np.int64)
    for row, i in enumerate(idx):
        s = docs[i]
        if len(s) > length:
            start = np.random.randint(0, len(s) - length + 1) if train_mode else 0
            s = s[start:start + length]
        x[row, :len(s)] = s
    return torch.from_numpy(x).to(device)

# alt 1, learned embeddings averaged then fed to a classifier
class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_id)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mask = (x != pad_id).float()
        h = self.emb(x)
        h = h * mask.unsqueeze(-1)
        h = h.sum(1)
        denominator = mask.sum(1).clamp(min=1).unsqueeze(-1)
        h = h / denominator
        return self.classifier(h).squeeze(1)

def run_embedding_classifier(epochs=10, batch_size=64):
    oof = np.zeros(len(docs))
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        torch.manual_seed(fold_no)
        np.random.seed(fold_no)
        model = EmbeddingClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()
            order = np.random.permutation(train_idx)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                x = make_batch(docs, idx, True)
                target = torch.tensor(y[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), target)
                loss.backward()
                optimizer.step()

        model.eval()
        val_prob = []
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                idx = val_idx[start:start + batch_size]
                x = make_batch(docs, idx, False)
                prob = torch.sigmoid(model(x))
                val_prob.append(prob.cpu().numpy())
        oof[val_idx] = np.concatenate(val_prob)
        print(f"embedding fold {fold_no}/5 done")

    print_results("alternative 1: learned embeddings + neural classifier", y, oof)
    return oof

# alt 2, bidirectional lstm over token embeddings
class LSTMClassifier(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mask = (x != pad_id)
        h = self.emb(x)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(1)

def run_lstm_classifier(epochs=10, batch_size=64):
    oof = np.zeros(len(docs))
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        torch.manual_seed(fold_no)
        np.random.seed(fold_no)
        model = LSTMClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            model.train()
            order = np.random.permutation(train_idx)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                x = make_batch(docs, idx, True)
                target = torch.tensor(y[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), target)
                loss.backward()
                optimizer.step()

        model.eval()
        val_prob = []
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                idx = val_idx[start:start + batch_size]
                x = make_batch(docs, idx, False)
                prob = torch.sigmoid(model(x))
                val_prob.append(prob.cpu().numpy())
        oof[val_idx] = np.concatenate(val_prob)
        print(f"lstm fold {fold_no}/5 done")

    print_results("alternative 2: lstm", y, oof)
    return oof

# language model features, hashes k-grams so counting is fast with np.unique
def ngram_keys(s, k):
    n = len(s) - k + 1
    keys = np.zeros(n, dtype=np.int64)
    for j in range(k):
        keys = keys * vocab_size + s[j:j + n]
    return keys

def count_ngrams(docs, k):
    keys = np.concatenate([ngram_keys(s, k) for s in docs if len(s) >= k])
    grams, counts = np.unique(keys, return_counts=True)
    ctx, start = np.unique(grams // vocab_size, return_index=True)
    ctx_counts = np.add.reduceat(counts, start)
    return grams, counts, ctx, ctx_counts

def build_lm(docs):
    uni = np.zeros(vocab_size)
    for s in docs:
        np.add.at(uni, s, 1)
    return {"uni": uni, "total": uni.sum(), "bi": count_ngrams(docs, 2), "tri": count_ngrams(docs, 3)}

def lookup(sorted_keys, values, query):
    idx = np.searchsorted(sorted_keys, query)
    idx[idx == len(sorted_keys)] = len(sorted_keys) - 1
    return np.where(sorted_keys[idx] == query, values[idx], 0)

def cond_prob(table, gram_keys, ctx_keys):
    grams, counts, ctx, ctx_counts = table
    num = lookup(grams, counts, gram_keys)
    den = lookup(ctx, ctx_counts, ctx_keys)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)

def token_logprobs(lm, s):
    p1 = (lm["uni"][s] + 1) / (lm["total"] + vocab_size)
    p2 = cond_prob(lm["bi"], ngram_keys(s, 2), s[:-1])
    p = p1.copy()
    if len(s) < 3:
        p[1:] = 0.4 * p1[1:] + 0.6 * p2
    else:
        p3 = cond_prob(lm["tri"], ngram_keys(s, 3), ngram_keys(s[:-1], 2))
        p[1] = 0.4 * p1[1] + 0.6 * p2[0]
        p[2:] = 0.25 * p1[2:] + 0.30 * p2[1:] + 0.50 * p3
    return np.log(p)

def lm_features(lm_a, lm_b, docs):
    output = np.zeros((len(docs), 11))
    for i, s in enumerate(docs):
        if len(s) < 2:
            continue
        a = token_logprobs(lm_a, s)
        b = token_logprobs(lm_b, s)
        d = b - a
        half = len(d) // 2
        output[i] = [d.mean(), d.std(), np.median(d), (d > 0).mean(),
                     d[:half].mean(), d[half:].mean(), d.sum(),
                     a.mean(), b.mean(), a.std(), b.std()]
    return output

def lm_oof(docs, y, folds):
    output = np.zeros((len(docs), 11))
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        lm_a = build_lm([docs[i] for i in train_idx if y[i] == 0])
        lm_b = build_lm([docs[i] for i in train_idx if y[i] == 1])
        output[val_idx] = lm_features(lm_a, lm_b, [docs[i] for i in val_idx])
        print(f"lm fold {fold_no}/5 done")
    return output

# tfidf + svm stage-i, 3 vectorizers (uni/bi/tri) each give one score column
def svm_oof(docs, y, folds):
    n = len(docs)
    strings = [" ".join(map(str, s)) for s in docs]
    settings = {
        "bi": {"ngram_range": (1, 2), "min_df": 2},
        "tri": {"ngram_range": (1, 3), "min_df": 3, "max_features": 1_500_000},
        "uni": {"ngram_range": (1, 1), "min_df": 1},
    }
    output = np.zeros((n, len(settings)))
    for j, (name, parameters) in enumerate(settings.items()):
        vectorizer = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, lowercase=False,
                                      dtype=np.float32, **parameters)
        X = vectorizer.fit_transform(strings).tocsr()
        for train_idx, val_idx in folds:
            model = LinearSVC(C=1.2, max_iter=100000)
            model.fit(X[train_idx], y[train_idx])
            output[val_idx, j] = model.decision_function(X[val_idx])
        print(f"svm {name} done")
    return output

# handcrafted stats, length/diversity/repetition/rarity of tokens
def stat_features(docs, freq_log):
    features = []
    for s in docs:
        n = len(s)
        if n < 3:
            features.append([0.0] * 9)
            continue
        uniq, counts = np.unique(s, return_counts=True)
        p = counts / n
        entropy = -(p * np.log(p)).sum()
        bigram_rep = 1 - len(np.unique(ngram_keys(s, 2))) / (n - 1)
        trigram_rep = 1 - len(np.unique(ngram_keys(s, 3))) / (n - 2)
        seen50 = np.mean([s[t] in s[max(0, t - 50):t] for t in range(1, n)])
        lf = freq_log[s]
        half = n // 2
        features.append([n, len(uniq), len(uniq) / n, entropy, bigram_rep, trigram_rep,
                          seen50, lf.mean(), lf[half:].mean() - lf[:half].mean()])
    return np.array(features, dtype=np.float64)

def build_stage1():
    print()
    print("=" * 60)
    print("building stage-i features")
    print("=" * 60)

    svm_features = svm_oof(docs, y, folds)
    lm_features_oof = lm_oof(docs, y, folds)

    freq = np.zeros(vocab_size)
    for s in docs:
        np.add.at(freq, s, 1)
    freq_log = np.log1p(freq)
    stats = stat_features(docs, freq_log)

    print("training cnn to obtain cnn stage-i signal...")
    cnn_features = run_cnn_oof()

    z = np.hstack([svm_features, lm_features_oof, stats, cnn_features[:, None]])
    print("stage-i matrix:", z.shape)
    return z

# cnn stage-i, 3 conv widths + max/mean pool each
class CNN(nn.Module):
    def __init__(self, emb_dim=128, channels=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(0.25)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = nn.Sequential(
            nn.Linear(2 * channels * 3 + 1, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mask = (x != pad_id).unsqueeze(1)
        h = self.drop(self.emb(x)).transpose(1, 2)
        features = []
        for conv in self.convs:
            z = torch.relu(conv(h))
            features.append(z.masked_fill(~mask, -1e4).max(2).values)
            features.append((z * mask).sum(2) / mask.sum(2).clamp(min=1))
        length = torch.log1p(mask.sum(2).float()) / 7.0
        return self.head(torch.cat(features + [length], 1)).squeeze(1)

@torch.no_grad()
def cnn_predict(model, docs, idx, batch_size=128):
    model.eval()
    probabilities = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        x = make_batch(docs, batch_idx, False)
        probabilities.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(probabilities)

def run_cnn_oof(epochs=8, batch_size=64):
    oof = np.zeros(len(docs))
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        torch.manual_seed(fold_no)
        np.random.seed(fold_no)
        model = CNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        best_acc = -1
        best_val = None

        for epoch in range(epochs):
            model.train()
            order = np.random.permutation(train_idx)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                x = make_batch(docs, idx, True)
                target = torch.tensor(y[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), target)
                loss.backward()
                optimizer.step()

            val_prob = cnn_predict(model, docs, val_idx)
            val_pred = (val_prob >= 0.5).astype(int)
            acc = accuracy_score(y[val_idx], val_pred)
            if acc > best_acc:
                best_acc = acc
                best_val = val_prob.copy()

        oof[val_idx] = best_val
        print(f"cnn fold {fold_no}/5 best acc = {best_acc:.4f}")
    return oof

# alt 3, plain logistic regression on top of stage-i features
def run_linear_stage2(z):
    oof = np.zeros(len(y))
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        model = LogisticRegression(C=1.0, max_iter=5000)
        model.fit(z[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(z[val_idx])[:, 1]
        print(f"linear stage-ii fold {fold_no}/5 done")
    print_results("alternative 3: linear stage-ii", y, oof)
    return oof

# alt 4, small mlp on top of stage-i features
class Stage2NN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def run_neural_stage2(z, epochs=30):
    oof = np.zeros(len(y))
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        torch.manual_seed(fold_no)
        model = Stage2NN(z.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

        x_train = torch.tensor(z[train_idx], dtype=torch.float32, device=device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
        x_val = torch.tensor(z[val_idx], dtype=torch.float32, device=device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(model(x_val)).cpu().numpy()
        oof[val_idx] = probability
        print(f"neural stage-ii fold {fold_no}/5 done")

    print_results("alternative 4: neural network stage-ii", y, oof)
    return oof

def main():
    print()
    print("=" * 60)
    print("alternative 1")
    print("=" * 60)
    run_embedding_classifier()

    print()
    print("=" * 60)
    print("alternative 2")
    print("=" * 60)
    run_lstm_classifier()

    print()
    print("=" * 60)
    print("building features for stage-ii alternatives")
    print("=" * 60)
    z = build_stage1()

    print()
    print("=" * 60)
    print("alternative 3")
    print("=" * 60)
    run_linear_stage2(z)

    print()
    print("=" * 60)
    print("alternative 4")
    print("=" * 60)
    run_neural_stage2(z)

if __name__ == "__main__":
    main()