"""Step 2 -- CNN classifiers on the shared 25 cross-validation folds (folds.json).

    python3 src/run_cnn.py <model> [first_fold] [last_fold] [budget_s]
    python3 src/run_cnn.py collect
    model: scratch_orig | scratch_improved | resnet18

scratch_*  : compact CNN, Conv(32,3)-ReLU-MaxPool / Conv(64,3)-ReLU-MaxPool /
             Conv(128,3)-ReLU / Flatten / Dense(64)-ReLU / Dense(4), 128x128 RGB
             in [0,1], Glorot-uniform initialisation, Adam lr 1e-3, batch 8.
  scratch_orig     : 5 epochs on the whole training fold, no augmentation.
  scratch_improved : stratified 20% validation split of the training fold,
                     random h-flip + brightness jitter, <=60 epochs, early
                     stopping on validation loss (patience 8), best weights restored.
resnet18   : ImageNet-pretrained ResNet-18, conv1..layer3 frozen (eval mode),
             layer4 + new FC fine-tuned (Adam 1e-4, batch 8), 224x224 input with
             ImageNet normalisation; augmentation = random h-flip (frozen-stage
             activations of each image and its mirror are precomputed); same
             validation split, <=40 epochs, early stopping patience 6.
Each fold writes results/cnn/<model>_fold<k>.json (accuracy + test predictions);
`collect` merges them into results/cnn_all.json.
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(max(1, os.cpu_count()))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORIENT = os.environ.get("SCI_ORIENT", "exif")
SUFFIX = "" if ORIENT == "exif" else "_stored"
CACHE = os.path.join(ROOT, "cache" + SUFFIX)  # written by extract_features.py
RES = os.path.join(ROOT, "results", "cnn" + SUFFIX)
os.makedirs(RES, exist_ok=True)

FOLDS = json.load(open(os.path.join(HERE, "folds.json")))
Y = np.array(FOLDS["y"])
CLASSES = FOLDS["classes"]


def load_arrays():
    p = os.path.join(CACHE, "cnn_inputs.npz")
    if not os.path.exists(p):
        fs = sorted(f for f in os.listdir(CACHE) if f.startswith("img"))
        assert len(fs) == 68, len(fs)
        d = [np.load(os.path.join(CACHE, f)) for f in fs]
        np.savez(p, r128=np.stack([x["r128"] for x in d]), r224=np.stack([x["r224"] for x in d]),
                 labels=np.array([str(x["label"]) for x in d]))
    a = np.load(p)
    assert [CLASSES.index(l) for l in a["labels"]] == Y.tolist()
    return a


class KerasCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3); self.c2 = nn.Conv2d(32, 64, 3); self.c3 = nn.Conv2d(64, 128, 3)
        self.f1 = nn.Linear(128 * 28 * 28, 64); self.f2 = nn.Linear(64, 4)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x))
        return self.f2(F.relu(self.f1(x.flatten(1))))


def batches(n, bs, rng):
    p = rng.permutation(n)
    return [p[i:i + bs] for i in range(0, n, bs)]


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        loss = F.cross_entropy(out, y).item()
        pred = out.argmax(1)
    return loss, (pred == y).float().mean().item(), pred.numpy()


def augment_scratch(x, g):
    flip = torch.rand(len(x), generator=g) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    b = 1 + (torch.rand(len(x), 1, 1, 1, generator=g) * 0.4 - 0.2)
    return (x * b).clamp(0, 1)


def run_scratch(fold, improved, arr):
    seed = 42 + fold["fold"]
    torch.manual_seed(seed); rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    X = torch.tensor(arr["r128"]).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(Y)
    model = KerasCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    info = {}
    if not improved:
        tr = np.array(fold["train"])
        for ep in range(5):
            model.train()
            for b in batches(len(tr), 8, rng):
                idx = tr[b]
                opt.zero_grad(); F.cross_entropy(model(X[idx]), y[idx]).backward(); opt.step()
        info["epochs"] = 5
    else:
        tr, va = np.array(fold["inner_train"]), np.array(fold["inner_val"])
        best, best_state, bad, best_ep = 1e9, None, 0, 0
        for ep in range(1, 61):
            model.train()
            for b in batches(len(tr), 8, rng):
                idx = tr[b]
                opt.zero_grad(); F.cross_entropy(model(augment_scratch(X[idx], g)), y[idx]).backward(); opt.step()
            vl, va_acc, _ = evaluate(model, X[va], y[va])
            if vl < best - 1e-4:
                best, best_state, bad, best_ep = vl, copy.deepcopy(model.state_dict()), 0, ep
            else:
                bad += 1
                if bad >= 8:
                    break
        model.load_state_dict(best_state)
        info.update(best_epoch=best_ep, stopped_epoch=ep, best_val_loss=best)
    te = np.array(fold["test"])
    _, acc, pred = evaluate(model, X[te], y[te])
    return acc, pred, info


def resnet_trunk_feats(arr):
    p = os.path.join(CACHE, "resnet_trunk.pt")
    if os.path.exists(p):
        return torch.load(p)
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    X = (torch.tensor(arr["r224"]).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
    outs = []
    with torch.no_grad():
        for Xv in (X, X.flip(3)):
            fs = []
            for i in range(0, len(Xv), 8):
                h = m.maxpool(m.relu(m.bn1(m.conv1(Xv[i:i + 8]))))
                fs.append(m.layer3(m.layer2(m.layer1(h))))
            outs.append(torch.cat(fs))
    T = torch.stack(outs)  # [2, 68, 256, 14, 14]
    torch.save(T, p)
    return T


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.layer4 = m.layer4
        self.fc = nn.Linear(512, 4)

    def forward(self, t):
        return self.fc(F.adaptive_avg_pool2d(self.layer4(t), 1).flatten(1))


def run_resnet(fold, T):
    seed = 42 + fold["fold"]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y = torch.tensor(Y)
    model = Head()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tr, va = np.array(fold["inner_train"]), np.array(fold["inner_val"])
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    for ep in range(1, 41):
        model.train()
        for b in batches(len(tr), 8, rng):
            idx = tr[b]
            flip = rng.integers(0, 2, len(idx))
            t = T[flip, idx]
            opt.zero_grad(); F.cross_entropy(model(t), y[idx]).backward(); opt.step()
        vl, _, _ = evaluate(model, T[0, va], y[va])
        if vl < best - 1e-4:
            best, best_state, bad, best_ep = vl, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    te = np.array(fold["test"])
    _, acc, pred = evaluate(model, T[0, te], y[te])
    return acc, pred, dict(best_epoch=best_ep, stopped_epoch=ep, best_val_loss=best)


if __name__ == "__main__" and sys.argv[1] == "collect":
    import glob
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "*.json")))]
    json.dump(rs, open(os.path.join(ROOT, "results", f"cnn_all{SUFFIX}.json"), "w")); print(len(rs), "fold results collected"); sys.exit()

if __name__ == "__main__":
    name = sys.argv[1]
    a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    b = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else float("inf")
    t0 = time.time()
    arr = load_arrays()
    T = resnet_trunk_feats(arr) if name == "resnet18" else None
    for k in range(a, b + 1):
        out = os.path.join(RES, f"{name}_fold{k:02d}.json")
        if os.path.exists(out):
            continue
        if time.time() - t0 > budget:
            print("budget reached before fold", k); break
        f = FOLDS["folds"][k]; t1 = time.time()
        if name == "resnet18":
            acc, pred, info = run_resnet(f, T)
        else:
            acc, pred, info = run_scratch(f, name == "scratch_improved", arr)
        json.dump(dict(model=name, fold=k, acc=acc, test=f["test"], pred=pred.tolist(),
                       secs=time.time() - t1, **info), open(out, "w"))
        print(f"{name} fold {k}: acc={acc:.3f} ({time.time()-t1:.0f}s) {info}", flush=True)
    print("total", round(time.time() - t0), "s")
