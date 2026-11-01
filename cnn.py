"""
1D-CNN over token embeddings. Trains 5 folds using the SAME split as solution_v2.py
(StratifiedKFold(5, shuffle=True, random_state=42)), so out-of-fold predictions align.
Saves nn_oof.npy (n_train,) and nn_test.npy (n_test,). Feed them to solution_v2.py:
    python cnn.py --train train.json --test test.json
    python solution_v2.py ... --nn_oof nn_oof.npy --nn_test nn_test.npy
Runs fine on CPU (slow) or GPU / Colab (minutes).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

V = 18438
PAD = V
MAXLEN = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    raw = open(path).read().strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]


def batchify(docs, idx, train):
    L = min(MAXLEN, max(len(docs[i]) for i in idx))
    L = max(L, 8)
    x = np.full((len(idx), L), PAD, dtype=np.int64)
    for r, i in enumerate(idx):
        s = docs[i]
        if len(s) > L:
            st = np.random.randint(0, len(s) - L + 1) if train else 0
            s = s[st:st + L]
        x[r, :len(s)] = s
    return torch.from_numpy(x).to(dev)


class CNN(nn.Module):
    def __init__(self, d=128, ch=128):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d, padding_idx=PAD)
        self.drop = nn.Dropout(0.25)
        self.convs = nn.ModuleList([nn.Conv1d(d, ch, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = nn.Sequential(nn.Linear(2 * ch * 3 + 1, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        mask = (x != PAD)
        h = self.drop(self.emb(x)).transpose(1, 2)              # B,d,L
        m = mask.unsqueeze(1)
        feats = []
        for c in self.convs:
            z = torch.relu(c(h))
            feats.append(z.masked_fill(~m, -1e4).max(2).values)
            feats.append((z * m).sum(2) / m.sum(2).clamp(min=1))
        ln = torch.log1p(mask.sum(1, keepdim=True).float()) / 7.0
        return self.head(torch.cat(feats + [ln], 1)).squeeze(1)


@torch.no_grad()
def predict(model, docs, idx, bs=128):
    model.eval()
    out = []
    order = sorted(idx, key=lambda i: len(docs[i]))
    for b in range(0, len(order), bs):
        ii = order[b:b + bs]
        out.append(torch.sigmoid(model(batchify(docs, ii, False))).cpu().numpy())
    p = np.concatenate(out)
    res = np.zeros(len(idx))
    pos = {i: k for k, i in enumerate(idx)}
    for i, v in zip(order, p):
        res[pos[i]] = v
    return res


def main(a):
    tr, te = load(a.train), load(a.test)
    y = np.array([1 if d["label"] == "B" else 0 for d in tr])
    docs_tr = [np.asarray(d["text"], dtype=np.int64) for d in tr]
    docs_te = [np.asarray(d["text"], dtype=np.int64) for d in te]
    ntr = len(tr)
    oof = np.zeros(ntr)
    tst = np.zeros(len(te))
    te_idx = list(range(len(te)))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    for f, (i_tr, i_va) in enumerate(splits):
        torch.manual_seed(f)
        np.random.seed(f)
        model = CNN().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.epochs * (len(i_tr) // a.bs + 1))
        lossf = nn.BCEWithLogitsLoss()
        best, best_va, best_te = -1, None, None
        for ep in range(a.epochs):
            model.train()
            perm = np.random.permutation(i_tr)
            for b in range(0, len(perm), a.bs):
                ii = perm[b:b + a.bs]
                x = batchify(docs_tr, ii, True)
                t = torch.tensor(y[ii], dtype=torch.float32, device=dev)
                opt.zero_grad()
                loss = lossf(model(x), t)
                loss.backward()
                opt.step()
                sched.step()
            pv = predict(model, docs_tr, list(i_va))
            acc = accuracy_score(y[i_va], pv > .5)
            print(f"fold {f} ep {ep} val acc {acc:.4f}")
            if acc > best:
                best, best_va, best_te = acc, pv, predict(model, docs_te, te_idx)
        oof[i_va] = best_va
        tst += best_te / 5
        print(f"fold {f} best {best:.4f}")
    print(f"NN OOF acc = {accuracy_score(y, oof > .5):.4f}")
    np.save("nn_oof.npy", oof)
    np.save("nn_test.npy", tst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.json")
    ap.add_argument("--test", default="test.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    main(ap.parse_args())
