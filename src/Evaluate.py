import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_mnist

def evaluate_model():
    x_train, y_train, x_test, y_test = load_mnist()
    model = tf.keras.models.load_model("models/cnn_mnist_model.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    return cm

if __name__ == "__main__":
    evaluate_model()
