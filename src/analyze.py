"""Step 4 -- statistics, tables and figures. Combines results/traditional.json
and results/cnn_all.json (all on the same 25 folds): mean, SD, 95% CI,
corrected resampled t-test (Nadeau & Bengio, 2003) with Holm adjustment,
pooled confusion matrices, and the figures in figures/.

    python3 src/analyze.py      -> results/final.json, figures/*.png
"""
import json, itertools, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); os.chdir(ROOT)
FOLDS = json.load(open(os.path.join(HERE, "folds.json")))
y = np.array(FOLDS["y"]); classes = FOLDS["classes"]
N = json.load(open("results/traditional.json"))
C = json.load(open("results/cnn_all.json"))  # from run_cnn.py collect

M = {"DCT/JPEG": N["cv"]["jpeg"], "PRNU-inspired": N["cv"]["prnu"]}
for key, name in [("scratch_orig", "CNN (scratch, 5 ep.)"), ("scratch_improved", "CNN (scratch, aug.+ES)"),
                  ("resnet18", "ResNet-18 (ImageNet)")]:
    rs = sorted([r for r in C if r["model"] == key], key=lambda r: r["fold"])
    assert [r["fold"] for r in rs] == list(range(25))
    for r, f in zip(rs, FOLDS["folds"]):
        assert r["test"] == f["test"]
    M[name] = {"fold_acc": [r["acc"] for r in rs], "pred": [r["pred"] for r in rs],
               "best_epochs": [r.get("best_epoch") for r in rs]}

J = 25; n_te = np.mean([len(f["test"]) for f in FOLDS["folds"]]); n_tr = np.mean([len(f["train"]) for f in FOLDS["folds"]])


def summary(a):
    a = np.asarray(a); m = a.mean(); s = a.std(ddof=1)
    h = stats.t.ppf(0.975, J - 1) * s / np.sqrt(J)
    return dict(mean=m, std=s, ci=[m - h, m + h])


def corrected_t(a, b):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(a) - np.asarray(b)
    v = d.var(ddof=1)
    if v == 0:
        return np.inf, 0.0
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v)
    return t, 2 * stats.t.sf(abs(t), J - 1)


out = {"methods": {}, "pairs": []}
for name, r in M.items():
    s = summary(r["fold_acc"])
    yt = np.concatenate([y[f["test"]] for f in FOLDS["folds"]]); yp = np.concatenate(r["pred"])
    cm = confusion_matrix(yt, yp, labels=range(4))
    s.update(pooled_acc=float((yt == yp).mean()),
             macro_prec=float(precision_score(yt, yp, average="macro", zero_division=0)),
             macro_rec=float(recall_score(yt, yp, average="macro", zero_division=0)),
             per_class_recall=(cm.diagonal() / cm.sum(1)).tolist(), cm=cm.tolist())
    if r.get("best_epochs"):
        be = [e for e in r["best_epochs"] if e]
        if be:
            s["best_epoch_median"] = float(np.median(be))
    out["methods"][name] = s
    print(f"{name:26s} {s['mean']:.3f} ± {s['std']:.3f}  CI[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]  "
          f"P={s['macro_prec']:.2f} R={s['macro_rec']:.2f} rec={np.round(s['per_class_recall'],2)}")

names = list(M)
raw = []
for a, b in itertools.combinations(names, 2):
    t, p = corrected_t(M[a]["fold_acc"], M[b]["fold_acc"])
    try:
        w = stats.wilcoxon(M[a]["fold_acc"], M[b]["fold_acc"]).pvalue
    except ValueError:
        w = 1.0
    raw.append(dict(a=a, b=b, diff=float(np.mean(M[a]["fold_acc"]) - np.mean(M[b]["fold_acc"])), t=float(t), p=float(p), p_wilcoxon=float(w)))
# Holm correction over the 10 comparisons
order = np.argsort([r["p"] for r in raw]); m = len(raw); prev = 0
for rank, i in enumerate(order):
    adj = min(1.0, max(prev, (m - rank) * raw[i]["p"])); raw[i]["p_holm"] = adj; prev = adj
for r in raw:
    r["sig"] = r["p_holm"] < 0.05
    print(f"{r['a']:24s} vs {r['b']:24s} diff={r['diff']:+.3f} p_corr={r['p']:.2e} holm={r['p_holm']:.2e} wil={r['p_wilcoxon']:.2e} {'*' if r['sig'] else ''}")
out["pairs"] = raw
out["format"] = {"cv_native": {k: N["cv"][k]["mean"] for k in ["jpeg", "prnu"]},
                 "cv_harmonized": {k: summary(N["cv"][k + "_harmonized"]["fold_acc"]) for k in ["jpeg", "prnu"]},
                 "pairs": {p: {k: summary(v["fold_acc"]) for k, v in d.items()} for p, d in N["pairs"].items()}}
for k in ["jpeg", "prnu"]:
    t, p = corrected_t(N["cv"][k]["fold_acc"], N["cv"][k + "_harmonized"]["fold_acc"])
    out["format"]["native_vs_harmonized_" + k] = dict(t=float(t), p=float(p))
    print("native vs harmonized", k, round(N["cv"][k]["mean"], 3), round(N["cv"][k + "_harmonized"]["mean"], 3), "p=", round(p, 4))
out["shapes"] = N["shapes"]
json.dump(out, open("results/final.json", "w"), indent=1, default=float)

# ---------------- figures ----------------
import os
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9})
cols = ["#4C78A8", "#E45756", "#B0B0B0", "#72B7B2", "#54A24B"]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
means = [out["methods"][n]["mean"] for n in names]
lo = [means[i] - out["methods"][n]["ci"][0] for i, n in enumerate(names)]
hi = [out["methods"][n]["ci"][1] - means[i] for i, n in enumerate(names)]
ax.bar(range(5), means, yerr=[lo, hi], capsize=4, color=cols)
for i, n in enumerate(names):
    ax.scatter(np.full(25, i) + np.random.default_rng(i).uniform(-0.25, 0.25, 25), M[n]["fold_acc"], s=6, c="k", alpha=0.35, zorder=3)
ax.axhline(0.25, ls="--", c="gray", lw=1); ax.text(-0.45, 0.265, "chance (0.25)", fontsize=7, ha="left", color="gray")
ax.set_xticks(range(5)); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7.5)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig("figures/accuracy_cv.png", dpi=300); plt.close(fig)

short = ["iPh13PM", "iPh17P", "Redmi", "S24"]
sel = ["DCT/JPEG", "PRNU-inspired", "CNN (scratch, aug.+ES)", "ResNet-18 (ImageNet)"]
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0))
for ax, n in zip(axes.flat, sel):
    cm = np.array(out["methods"][n]["cm"], float); cm /= cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=7, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title(n, fontsize=8.5); ax.set_xlabel("Predicted", fontsize=7.5)
[a.set_ylabel("True", fontsize=7.5) for a in axes[:,0]]
fig.tight_layout(); fig.savefig("figures/confusion.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 3.2))
groups = [("4-class", out["format"]["cv_native"], {k: v["mean"] for k, v in out["format"]["cv_harmonized"].items()})]
labels_g, nat_j, har_j, nat_p, har_p = [], [], [], [], []
labels_g.append("All 4 devices"); nat_j.append(N["cv"]["jpeg"]["mean"]); har_j.append(N["cv"]["jpeg_harmonized"]["mean"])
nat_p.append(N["cv"]["prnu"]["mean"]); har_p.append(N["cv"]["prnu_harmonized"]["mean"])
for p, d in N["pairs"].items():
    labels_g.append("iPhone 13 PM vs 17 Pro\n(both HEIC)" if "iphone" in p else "Redmi vs S24\n(both JPEG)")
    nat_j.append(d["jpeg"]["mean"]); har_j.append(d["jpeg_harmonized"]["mean"]); nat_p.append(d["prnu"]["mean"]); har_p.append(d["prnu_harmonized"]["mean"])
x = np.arange(3); w = 0.2
ax.bar(x - 1.5 * w, nat_j, w, color="#4C78A8", label="DCT/JPEG, native")
ax.bar(x - 0.5 * w, har_j, w, color="#9ECAE9", label="DCT/JPEG, re-encoded q90")
ax.bar(x + 0.5 * w, nat_p, w, color="#E45756", label="PRNU-insp., native")
ax.bar(x + 1.5 * w, har_p, w, color="#FF9D98", label="PRNU-insp., re-encoded q90")
ax.plot([-0.45, 0.45], [0.25, 0.25], "--", c="gray", lw=1); ax.plot([0.55, 2.45], [0.5, 0.5], "--", c="gray", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels_g, fontsize=7.5); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
ax.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout(); fig.savefig("figures/format.png", dpi=300); plt.close(fig)
print("figures saved")
