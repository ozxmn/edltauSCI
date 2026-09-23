"""Step 1 -- per-image feature extraction from the native image files
(camera JPEG for the Android devices, HEIC decoded with pillow-heif for the
iPhones). For every image it caches: the 98-dim DCT/JPEG feature vector, the
50,000-dim PRNU-inspired residual, both features again after re-encoding the
image as JPEG (quality 90) with a common encoder, and 128x128 / 224x224 resized
RGB copies for the CNNs. Resumable: one .npz per image in cache/, and an
optional time budget (seconds) as first argument; re-run until it reports
done 68/68, after which cache/features.npz is written.

    python3 src/extract_features.py [budget_s]
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image, ImageOps
import pillow_heif
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dataset root with one sub-folder per device (Zenodo record 17640511)
DATA = os.environ.get("SCI_DATA", os.path.join(ROOT, "data"))
# Images are rotated upright according to their EXIF orientation tag (only the
# Samsung files carry a non-trivial tag). SCI_ORIENT=stored keeps the stored
# pixel orientation instead (orientation control experiment).
ORIENT = os.environ.get("SCI_ORIENT", "exif")
CACHE = os.path.join(ROOT, "cache" if ORIENT == "exif" else "cache_stored")
os.makedirs(CACHE, exist_ok=True)
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")


def load_image(path):
    if path.lower().endswith("heic"):
        heif = pillow_heif.read_heif(path)
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
    else:
        img = Image.open(path)
        if ORIENT == "exif":
            img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def _dct_basis_8():
    k = np.arange(8).reshape(-1, 1); n = np.arange(8).reshape(1, -1)
    D = np.cos((2 * n + 1) * k * np.pi / 16.0)
    a = np.full(8, 0.5); a[0] = 1.0 / np.sqrt(8)
    return D * a.reshape(-1, 1)
DCT_D = _dct_basis_8()


def jpeg_feat(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = g.shape; h8, w8 = h - h % 8, w - w % 8
    g = g[:h8, :w8].astype(np.float32)
    nh, nw = h8 // 8, w8 // 8
    ac_sum = np.zeros(49); ac_sq = np.zeros(49); n = 0
    # row-chunked to bound memory on 12-24 MP images
    for r0 in range(0, nh, 64):
        blk = g[r0 * 8:(r0 + 64) * 8].reshape(-1, 8, nw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8).astype(np.float64)
        c = np.einsum('ki,nij,lj->nkl', DCT_D, blk, DCT_D, optimize=True)[:, 1:, 1:].reshape(len(blk), -1)
        ac_sum += c.sum(0); ac_sq += (c ** 2).sum(0); n += len(c)
    mean = ac_sum / n
    var = ac_sq / n - mean ** 2
    return np.concatenate([mean, var]).astype(np.float64)


def prnu_feat(img):
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    noise = gray - gaussian_filter(gray, 1)
    lm = gaussian_filter(noise, 5)
    lv = gaussian_filter(noise ** 2, 5) - lm ** 2
    den = (noise - lm) * (lv / (lv + 0.01))
    return (noise - den).flatten()[:50000].astype(np.float32)


def harmonize(img, q=90):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def items():
    out = []
    for c in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, c)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(("jpg", "jpeg", "png", "heic")):
                out.append((c, f, os.path.join(d, f)))
    return out


if __name__ == "__main__":
    t0 = time.time()
    its = items()
    done = 0
    for i, (c, f, p) in enumerate(its):
        outp = os.path.join(CACHE, f"img{i:03d}.npz")
        if os.path.exists(outp):
            done += 1
            continue
        if time.time() - t0 > BUDGET:
            break
        img = load_image(p)
        hm = harmonize(img)
        np.savez(outp, label=c, fname=f, shape=np.array(img.shape),
                 jpeg=jpeg_feat(img), prnu=prnu_feat(img),
                 jpeg_h=jpeg_feat(hm), prnu_h=prnu_feat(hm),
                 r128=cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR),
                 r224=cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR))
        del img, hm
        done += 1
    print(f"done {done}/{len(its)} in {time.time()-t0:.0f}s")
    if done == len(its):  # all images processed: write the feature file used by run_traditional.py
        d = [np.load(os.path.join(CACHE, f"img{i:03d}.npz")) for i in range(len(its))]
        np.savez_compressed(os.path.join(CACHE, "features.npz"),
                            labels=np.array([str(x["label"]) for x in d]), fnames=np.array([str(x["fname"]) for x in d]),
                            shapes=np.stack([x["shape"] for x in d]), jpeg=np.stack([x["jpeg"] for x in d]),
                            prnu=np.stack([x["prnu"] for x in d]), jpeg_h=np.stack([x["jpeg_h"] for x in d]),
                            prnu_h=np.stack([x["prnu_h"] for x in d]))
        print("wrote", os.path.join(CACHE, "features.npz"))
