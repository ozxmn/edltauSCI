import os
import numpy as np
from PIL import Image
import pillow_heif
import cv2

def load_image(path):
    ext = path.lower().split(".")[-1]
    if ext == "heic":
        heif = pillow_heif.read_heif(path)
        img = Image.frombytes(
            heif.mode,
            heif.size,
            heif.data,
            "raw"
        )
    else:
        img = Image.open(path)

    img = img.convert("RGB")
    return np.array(img)

def collect_dataset(root):
    labels = []
    images = []

    for label_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, label_name)
        if not os.path.isdir(class_dir):
            continue
        
        for f in os.listdir(class_dir):
            if not f.lower().endswith(("jpg","jpeg","png","heic")):
                continue
            
            img = load_image(os.path.join(class_dir, f))
            images.append(img)
            labels.append(label_name)

    return images, labels
