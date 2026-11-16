import json, re, glob
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def find_file(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"./**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found. Add the competition data as Input.")
    return hits[0]


train = pd.read_csv(find_file("train.csv"))
test = pd.read_csv(find_file("test.csv"))
print("Train:", train.shape, "| Test:", test.shape)


# ---------- Parsing: columns are JSON lists of strings ----------
def parse(s):
    try:
        v = json.loads(s)
        return " ".join(x for x in v if isinstance(x, str))
    except Exception:
        return str(s)


for df in (train, test):
    for c in ["prompt", "response_a", "response_b"]:
        df[c] = df[c].fillna("").map(parse)

y = train[["winner_model_a", "winner_model_b", "winner_tie"]].values.argmax(1)

# ---------- Handcrafted features ----------
REFUSAL = re.compile(r"\b(i can't|i cannot|i'm sorry|i am sorry|as an ai|i'm unable|i am unable)\b", re.I)


def text_stats(t):
    words = t.split()
    nw = len(words)
    return [
        len(t), nw, t.count("\n"), t.count("```"),
        len(re.findall(r"^\s*[-*•]\s", t, re.M)),      # bullets
        len(re.findall(r"^\s*\d+[.)]\s", t, re.M)),    # numbered lists
        t.count("**"), len(re.findall(r"^#+\s", t, re.M)),
        len(set(w.lower() for w in words)) / (nw + 1),  # lexical diversity
        sum(c.isdigit() for c in t) / (len(t) + 1),
        int(bool(REFUSAL.search(t))),
        t.count("?"), t.count("!"),
    ]


def overlap(p, r):
    pw, rw = set(p.lower().split()), set(r.lower().split())
    return len(pw & rw) / (len(pw) + 1)


def stats_matrix(df, swap=False):
    rows = []
    for p, a, b in zip(df.prompt, df.response_a, df.response_b):
        if swap:
            a, b = b, a
        sa, sb = np.array(text_stats(a), float), np.array(text_stats(b), float)
        rows.append(np.concatenate([
            sa, sb, sa - sb,
            [len(p), len(p.split()), overlap(p, a), overlap(p, b), overlap(p, a) - overlap(p, b)],
        ]))
    return np.vstack(rows)


# ---------- TF-IDF difference -> SVD ----------
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
tfidf.fit(pd.concat([train.response_a, train.response_b, test.response_a, test.response_b]))
svd = TruncatedSVD(n_components=64, random_state=42)


def tfidf_diff(df, swap=False):
    A, B = tfidf.transform(df.response_a), tfidf.transform(df.response_b)
    return (B - A) if swap else (A - B)


svd.fit(vstack([tfidf_diff(train), tfidf_diff(train, swap=True)]))


def build(df, swap=False):
    return np.hstack([stats_matrix(df, swap), svd.transform(tfidf_diff(df, swap))])


def swap_labels(lbl):
    # A <-> B, tie stays tie
    return np.where(lbl == 0, 1, np.where(lbl == 1, 0, 2))


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=400, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )


def predict_symmetric(model, df):
    p1 = model.predict_proba(build(df))
    p2 = model.predict_proba(build(df, swap=True))[:, [1, 0, 2]]  # unswap A/B columns
    return (p1 + p2) / 2


# ---------- Holdout check ----------
tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.2, stratify=y, random_state=42)
tr, va = train.iloc[tr_idx], train.iloc[va_idx]
X_tr = np.vstack([build(tr), build(tr, swap=True)])
y_tr = np.concatenate([y[tr_idx], swap_labels(y[tr_idx])])

m = make_model().fit(X_tr, y_tr)
p_va = np.clip(predict_symmetric(m, va), 1e-4, 1 - 1e-4)
print(f"Holdout log loss: {log_loss(y[va_idx], p_va):.5f}")
print(f"Prior-only baseline: {log_loss(y[va_idx], np.tile(np.bincount(y) / len(y), (len(va_idx), 1))):.5f}")

# ---------- Fit on all data, predict test ----------
X_all = np.vstack([build(train), build(train, swap=True)])
y_all = np.concatenate([y, swap_labels(y)])
final = make_model().fit(X_all, y_all)
p_test = np.clip(predict_symmetric(final, test), 1e-4, 1 - 1e-4)
p_test /= p_test.sum(1, keepdims=True)

sub = pd.DataFrame({
    "id": test["id"],
    "winner_model_a": p_test[:, 0],
    "winner_model_b": p_test[:, 1],
    "winner_tie": p_test[:, 2],
})
sub.to_csv("submission.csv", index=False)
print(sub.head())