import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def wiener_denoise(img):
    kernel_size = 5
    local_mean = gaussian_filter(img, kernel_size)
    local_var = gaussian_filter(img**2, kernel_size) - local_mean**2
    noise = (img - local_mean) * (local_var / (local_var + 0.01))
    return noise

def extract_prnu(img):
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    noise = gray - gaussian_filter(gray, 1)
    prnu = noise - wiener_denoise(noise)
    return prnu.flatten()


class PRNUModel:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear"))
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
