import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data_loader import load_mnist

def plot_confusion_matrix():
    x_train, y_train, x_test, y_test = load_mnist()
    model = tf.keras.models.load_model("models/cnn_mnist_model.h5")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    plot_confusion_matrix()
