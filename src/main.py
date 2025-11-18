from utils import collect_dataset
from jpeg_artifacts import jpeg_artifact_features, JPEGArtifactModel
from prnu import extract_prnu, PRNUModel
from cnn_model import CNNModel

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATA_DIR = "data/"

print("Loading images...")
images, labels = collect_dataset(DATA_DIR)
device_names = sorted(list(set(labels)))

print("Extracting JPEG features...")
X_jpeg = np.array([jpeg_artifact_features(img) for img in images])
pca = PCA(n_components=20)
X_jpeg = pca.fit_transform(X_jpeg)

X_jpeg_train, X_jpeg_test, y_jpeg_train, y_jpeg_test = train_test_split(
    X_jpeg, labels, test_size=0.3, random_state=42, stratify=labels
)

jpeg_model = JPEGArtifactModel()
jpeg_model.train(X_jpeg_train, y_jpeg_train)
acc_jpeg = jpeg_model.score(X_jpeg_test, y_jpeg_test)
print("JPEG Artifact accuracy:", acc_jpeg)

print("Extracting PRNU features...")
X_prnu = np.array([extract_prnu(img) for img in images])
X_prnu = X_prnu[:, :50000]

X_prnu_train, X_prnu_test, y_prnu_train, y_prnu_test = train_test_split(
    X_prnu, labels, test_size=0.3, random_state=42, stratify=labels
)

prnu_model = PRNUModel()
prnu_model.train(X_prnu_train, y_prnu_train)
acc_prnu = prnu_model.score(X_prnu_test, y_prnu_test)
print("PRNU accuracy:", acc_prnu)

print("Training CNN...")
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels
)

cnn = CNNModel(img_size=128)
cnn.train(X_train, y_train, epochs=5)
acc_cnn = cnn.score(X_test, y_test)
print("CNN accuracy:", acc_cnn)

methods = ["JPEG", "PRNU", "CNN"]
accs = [acc_jpeg, acc_prnu, acc_cnn]

plt.figure(figsize=(6, 4))
plt.bar(methods, accs, color=["skyblue", "salmon", "lightgreen"])
plt.ylabel("Accuracy")
plt.title("Method comparison")
plt.ylim(0, 1)
plt.show()

def plot_conf_matrix(y_true, y_pred, title, device_names):
    cm = confusion_matrix(y_true, y_pred, labels=device_names)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=device_names, yticklabels=device_names, cbar=True)
    plt.xlabel("Predicted device")
    plt.ylabel("Real device")
    plt.title(title)
    plt.show()

y_jpeg_pred = jpeg_model.predict(X_jpeg_test)
plot_conf_matrix(y_jpeg_test, y_jpeg_pred, "JPEG Artifact Analysis", device_names)

y_prnu_pred = prnu_model.predict(X_prnu_test)
plot_conf_matrix(y_prnu_test, y_prnu_pred, "PRNU Analysis", device_names)

Xp_test = cnn.preprocess(X_test)
y_cnn_pred = cnn.encoder.inverse_transform(np.argmax(cnn.model.predict(Xp_test), axis=1))
plot_conf_matrix(y_test, y_cnn_pred, "CNN Classifier", device_names)
