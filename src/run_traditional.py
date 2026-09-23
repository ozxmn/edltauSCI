"""Step 3 -- DCT/JPEG and PRNU-inspired classifiers on the shared 25
cross-validation folds (folds.json = RepeatedStratifiedKFold(5, 5,
random_state=42)). Standardisation, PCA and the SVM are fitted inside each
training fold. Also runs the format control experiments: same-format device
pairs and the common-JPEG-re-encoding condition.

    python3 src/run_traditional.py      -> results/traditional.json
"""
import json, os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
F = np.load(os.path.join(ROOT, "cache", "features.npz"))  # from extract_features.py
FOLDS = json.load(open(os.path.join(HERE, "folds.json")))
y = np.array(FOLDS["y"]); classes = FOLDS["classes"]
assert [classes.index(l) for l in F["labels"]] == y.tolist()
labels = F["labels"]


def jpeg_pipe(nc=20):
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=nc, random_state=42)), ("clf", SVC(kernel="rbf"))])


def prnu_pipe():
    return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear"))])


def run_folds(X, make, folds):
    accs, preds = [], []
    for f in folds:
        tr, te = np.array(f["train"]), np.array(f["test"])
        m = make().fit(X[tr], y[tr]); p = m.predict(X[te])
        accs.append(float((p == y[te]).mean())); preds.append(p.tolist())
    return accs, preds


out = {"n": int(len(y)), "shapes": {c: F["shapes"][y == i].tolist()[0] for i, c in enumerate(classes)}}

# 1) main 4-class repeated CV
out["cv"] = {}
for name, X, mk in [("jpeg", F["jpeg"], jpeg_pipe), ("prnu", F["prnu"], prnu_pipe),
                    ("jpeg_harmonized", F["jpeg_h"], jpeg_pipe), ("prnu_harmonized", F["prnu_h"], prnu_pipe)]:
    accs, preds = run_folds(X, mk, FOLDS["folds"])
    out["cv"][name] = {"fold_acc": accs, "pred": preds, "mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1))}
    print(name, round(np.mean(accs), 3), round(np.std(accs, ddof=1), 3))

# 2) within-format pairs (repeated stratified 5-fold, 5 repeats, on the 34 images of each pair)
out["pairs"] = {}
for pname, (ca, cb) in {"iphone13_vs_iphone17 (HEIC)": (0, 1), "redmi_vs_samsung (JPEG)": (2, 3)}.items():
    m = np.isin(y, [ca, cb]); yy = y[m]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    res = {}
    for name, X, mk in [("jpeg", F["jpeg"][m], lambda: jpeg_pipe(20)), ("prnu", F["prnu"][m], prnu_pipe),
                        ("jpeg_harmonized", F["jpeg_h"][m], lambda: jpeg_pipe(20)), ("prnu_harmonized", F["prnu_h"][m], prnu_pipe)]:
        accs = [float(mk().fit(X[tr], yy[tr]).score(X[te], yy[te])) for tr, te in cv.split(X, yy)]
        res[name] = {"fold_acc": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1))}
        print(pname, name, round(np.mean(accs), 3))
    out["pairs"][pname] = res

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "traditional.json"), "w"))
print("saved results/traditional.json")
