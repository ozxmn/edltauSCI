import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class CNNModel:
    def __init__(self, img_size=128):
        self.img_size = img_size
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input((self.img_size, self.img_size, 3)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(4, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def preprocess(self, imgs):
        arr = []
        for img in imgs:
            resized = tf.image.resize(img, (self.img_size, self.img_size))
            arr.append(resized / 255.0)
        return np.array(arr)

    def train(self, X, y, epochs=5):
        self.encoder = LabelEncoder()
        y_num = self.encoder.fit_transform(y)
        y_cat = to_categorical(y_num, num_classes=4)

        Xp = self.preprocess(X)
        self.model.fit(Xp, y_cat, epochs=epochs, batch_size=8)

    def score(self, X, y):
        Xp = self.preprocess(X)
        y_pred = self.encoder.inverse_transform(
            np.argmax(self.model.predict(Xp), axis=1)
        )
        return np.mean(y_pred == np.array(y))
