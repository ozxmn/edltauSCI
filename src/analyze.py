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
    """Mean, SD and 95% CI over the J folds. The interval uses the
    Nadeau-Bengio variance correction (1/J + n_test/n_train) so that it is
    consistent with the corrected t-test; the uncorrected interval is kept
    for reference as ci_uncorrected."""
    a = np.asarray(a); m = a.mean(); s = a.std(ddof=1); tq = stats.t.ppf(0.975, J - 1)
    h = tq * s * np.sqrt(1 / J + n_te / n_tr)
    h0 = tq * s / np.sqrt(J)
    return dict(mean=m, std=s, ci=[m - h, min(1.0, m + h)], ci_uncorrected=[m - h0, m + h0])


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
# Sized for the SNmult text width (345 pt = 4.79 in) and saved as vector PDF
# with embedded TrueType fonts; PNG copies are written for the README.
TEXTWIDTH = 345 / 72
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png", dpi=300)
    plt.close(fig)


# Fig. 1: processing pipeline
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.0))
steps = ["68 images,\n4 devices", "25 folds\n(repeated\nstratified\n5-fold CV)", "Method-\nspecific\npreprocessing",
         "Features and\nclassifier,\nfitted on\ntraining folds", "Predicted\ndevice for\nheld-out\nimages"]
n = len(steps); w, gap = 0.176, (0.985 - 5 * 0.176) / 4
for k, t in enumerate(steps):
    x0 = 0.0075 + k * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, 0.08), w, 0.84, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec="black", lw=0.7, transform=ax.transAxes))
    ax.text(x0 + w / 2, 0.5, t, ha="center", va="center", fontsize=7.5, linespacing=1.15, transform=ax.transAxes)
    if k < n - 1:
        ax.annotate("", xy=(x0 + w + gap - 0.004, 0.5), xytext=(x0 + w + 0.004, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", mutation_scale=8))
ax.axis("off"); fig.subplots_adjust(0.005, 0.02, 0.995, 0.98)
save(fig, "pipeline")

# Fig. 2: cross-validated accuracy
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.35))
means = [out["methods"][n_]["mean"] for n_ in names]
lo = [means[i] - out["methods"][n_]["ci"][0] for i, n_ in enumerate(names)]
hi = [out["methods"][n_]["ci"][1] - means[i] for i, n_ in enumerate(names)]
ax.bar(range(5), means, width=0.62, color="0.72", edgecolor="black", lw=0.6)
ax.errorbar(range(5), means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=0.8)
for i, n_ in enumerate(names):
    ax.scatter(np.full(25, i) + np.random.default_rng(i).uniform(-0.22, 0.22, 25), M[n_]["fold_acc"],
               s=5, c="black", alpha=0.45, lw=0, zorder=3)
ax.axhline(0.25, ls="--", c="0.35", lw=0.7)
ax.text(-0.45, 0.27, "chance level", fontsize=7, ha="left", color="0.3")
labels = ["DCT/JPEG", "PRNU-inspired", "CNN,\n5 epochs", "CNN, aug. +\nearly stopping", "ResNet-18\n(ImageNet)"]
ax.set_xticks(range(5)); ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3); save(fig, "accuracy_cv")

# Fig. 3: pooled confusion matrices (a-d)
short = ["iPh 13", "iPh 17", "Redmi", "S24"]
sel = ["DCT/JPEG", "PRNU-inspired", "CNN (scratch, aug.+ES)", "ResNet-18 (ImageNet)"]
fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.86))
for k, (ax, n_) in enumerate(zip(axes.flat, sel)):
    cm = np.array(out["methods"][n_]["cm"], float); cm /= cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=1.15)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.55 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_yticks(range(4)); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted device"); ax.set_ylabel("True device")
    ax.text(-0.42, 1.04, "abcd"[k], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
fig.tight_layout(pad=0.4, h_pad=1.2, w_pad=1.0); save(fig, "confusion")

# Fig. 4: format control experiments
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4))
groups = ["All four devices", "iPhone 13 Pro Max vs.\niPhone 17 Pro (HEIC)", "Redmi Note 10S vs.\nGalaxy S24 (JPEG)"]
vals = {k: [N["cv"][k]["mean"]] + [N["pairs"][p][k]["mean"] for p in N["pairs"]]
        for k in ["jpeg", "jpeg_harmonized", "prnu", "prnu_harmonized"]}
style = {"jpeg": dict(color="0.25", label="DCT/JPEG, native files"),
         "jpeg_harmonized": dict(color="white", hatch="////", label="DCT/JPEG, common re-encoding"),
         "prnu": dict(color="0.72", label="PRNU-inspired, native files"),
         "prnu_harmonized": dict(color="white", hatch="....", label="PRNU-inspired, common re-encoding")}
x = np.arange(3); w = 0.19
for off, k in zip([-1.5, -0.5, 0.5, 1.5], style):
    ax.bar(x + off * w, vals[k], w, edgecolor="black", lw=0.6, **style[k])
ax.plot([-0.45, 0.45], [0.25, 0.25], "--", c="0.35", lw=0.7); ax.plot([0.55, 2.45], [0.5, 0.5], "--", c="0.35", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.02); ax.set_ylabel("Accuracy")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(pad=0.3); save(fig, "format")
print("figures saved")
