import json, re, glob
import numpy as np
import pandas as pd
from scipy.sparse import vstack as svstack, hstack as shstack, csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ================= LOAD DATA =================
def find_file(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"./**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found. Add the competition data as Input.")
    return hits[0]


train = pd.read_csv(find_file("train.csv"))
test = pd.read_csv(find_file("test.csv"))
print("Train:", train.shape, "| Test:", test.shape)


def parse(s):
    try:
        return " ".join(x for x in json.loads(s) if isinstance(x, str))
    except Exception:
        return str(s)


for df in (train, test):
    for c in ["prompt", "response_a", "response_b"]:
        df[c] = df[c].fillna("").map(parse)

y = train[["winner_model_a", "winner_model_b", "winner_tie"]].values.argmax(1)
swap_labels = lambda l: np.where(l == 0, 1, np.where(l == 1, 0, 2))

# ================= FEATURES =================
REFUSAL = re.compile(r"\b(i can't|i cannot|i'm sorry|i am sorry|as an ai|i'm unable|i am unable)\b", re.I)


def text_stats(t):
    w = t.split(); nw = len(w)
    return [len(t), nw, t.count("\n"), t.count("```"),
            len(re.findall(r"^\s*[-*•]\s", t, re.M)), len(re.findall(r"^\s*\d+[.)]\s", t, re.M)),
            t.count("**"), len(re.findall(r"^#+\s", t, re.M)),
            len(set(x.lower() for x in w)) / (nw + 1), sum(c.isdigit() for c in t) / (len(t) + 1),
            int(bool(REFUSAL.search(t))), t.count("?"), t.count("!")]


def overlap(p, r):
    pw, rw = set(p.lower().split()), set(r.lower().split())
    return len(pw & rw) / (len(pw) + 1)


def stats_matrix(df, swap=False):
    rows = []
    for p, a, b in zip(df.prompt, df.response_a, df.response_b):
        if swap:
            a, b = b, a
        sa, sb = np.array(text_stats(a), float), np.array(text_stats(b), float)
        rows.append(np.concatenate([sa, sb, sa - sb,
                    [len(p), len(p.split()), overlap(p, a), overlap(p, b), overlap(p, a) - overlap(p, b)]]))
    return np.vstack(rows)


tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
tfidf.fit(pd.concat([train.response_a, train.response_b, test.response_a, test.response_b]))


def tfidf_diff(df, swap=False):
    A, B = tfidf.transform(df.response_a), tfidf.transform(df.response_b)
    return (B - A) if swap else (A - B)


svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(svstack([tfidf_diff(train), tfidf_diff(train, swap=True)]))

slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
scaler = StandardScaler().fit(slog(stats_matrix(train)))


def feat_lr(df, swap=False):
    S = scaler.transform(slog(stats_matrix(df, swap)))
    return shstack([csr_matrix(S), tfidf_diff(df, swap)]).tocsr()


def feat_gb(df, swap=False):
    return np.hstack([stats_matrix(df, swap), svd.transform(tfidf_diff(df, swap))])


# ================= RUNNER (swap augmentation + symmetric prediction) =================
tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.2, stratify=y, random_state=42)
tr, va = train.iloc[tr_idx], train.iloc[va_idx]


def run_model(make_model, featfn, name):
    stack = lambda a, b: svstack([a, b]) if issparse(a) else np.vstack([a, b])

    def fit(df, labels):
        X = stack(featfn(df, False), featfn(df, True))
        return make_model().fit(X, np.concatenate([labels, swap_labels(labels)]))

    def predict(m, df):
        p1 = m.predict_proba(featfn(df, False))
        p2 = m.predict_proba(featfn(df, True))[:, [1, 0, 2]]
        return np.clip((p1 + p2) / 2, 1e-4, 1 - 1e-4)

    p_va = predict(fit(tr, y[tr_idx]), va)
    print(f"[{name}] holdout log loss: {log_loss(y[va_idx], p_va):.5f}")
    p_te = predict(fit(train, y), test)
    return p_va, p_te


def write_sub(p, path):
    p = p / p.sum(1, keepdims=True)
    pd.DataFrame({"id": test["id"], "winner_model_a": p[:, 0],
                  "winner_model_b": p[:, 1], "winner_tie": p[:, 2]}).to_csv(path, index=False)
    print("Saved", path, p.shape)


# ================= CODE 1: LOGISTIC REGRESSION =================
p_va_lr, p_te_lr = run_model(
    lambda: LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs"), feat_lr, "1) LogReg")
write_sub(p_te_lr, "submission_logreg.csv")

# ================= CODE 2: GRADIENT BOOSTING =================
p_va_gb, p_te_gb = run_model(
    lambda: HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=400, max_depth=6, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=42),
    feat_gb, "2) HistGB")
write_sub(p_te_gb, "submission_hgb.csv")

# ================= CODE 3: BLEND =================
best_w, best_ll = 0.5, 9.0
for w in np.linspace(0, 1, 21):
    ll = log_loss(y[va_idx], w * p_va_gb + (1 - w) * p_va_lr)
    if ll < best_ll:
        best_w, best_ll = w, ll
print(f"[3) Blend] GB weight {best_w:.2f} | holdout log loss: {best_ll:.5f}")
write_sub(best_w * p_te_gb + (1 - best_w) * p_te_lr, "submission.csv")

# ================= SUMMARY =================
print("\n=== Holdout log loss summary ===")
print(f"LogReg : {log_loss(y[va_idx], p_va_lr):.5f}")
print(f"HistGB : {log_loss(y[va_idx], p_va_gb):.5f}")
print(f"Blend  : {best_ll:.5f}")