import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def jpeg_artifact_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    h8 = h - h % 8
    w8 = w - w % 8
    gray = gray[:h8, :w8]

    dct_blocks = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = np.float32(gray[i:i+8, j:j+8])
            dct_coeff = cv2.dct(block)
            dct_blocks.append(dct_coeff)

    dct_blocks = np.array(dct_blocks)

    # Feature vector: mean & var of AC coefficients
    ac = dct_blocks[:,1:,1:].reshape(len(dct_blocks), -1)
    feat = np.concatenate([ac.mean(axis=0), ac.var(axis=0)])

    return feat

class JPEGArtifactModel:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True))
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
