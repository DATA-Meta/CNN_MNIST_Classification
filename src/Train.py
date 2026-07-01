import tensorflow as tf
from data_loader import load_mnist
from model_builder import build_cnn_model

def train_model():
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_cnn_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )

    model.save("models/cnn_mnist_model.h5")
    return history

if __name__ == "__main__":
    train_model()
