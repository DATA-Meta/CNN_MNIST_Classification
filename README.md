🧠 Overview

A complete, modular, and production‑style deep learning project demonstrating how Convolutional Neural Networks (CNNs) learn spatial patterns and outperform classical ML models on image data.

This project includes:

Clean repository structure

Modular Python scripts

Training, evaluation, and visualization pipeline

Saved model + results

Academic report

🔖 Badges

   

🧩 Repository Structure

CNN_MNIST_Classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── week07_CNN_MNIST_Activity.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── models/
│   └── cnn_mnist_model.h5
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── sample_predictions.png
│   └── metrics.txt
│
└── reports/
    └── CNN_MNIST_Report.txt

🧱 Model Architecture Diagram

Input (28×28×1)
      ↓
Conv2D (32 filters, 3×3)
      ↓
MaxPooling2D (2×2)
      ↓
Conv2D (64 filters, 3×3)
      ↓
MaxPooling2D (2×2)
      ↓
Flatten
      ↓
Dense (64, ReLU)
      ↓
Dense (10, Softmax)
      ↓
Output (Digit Class)

🌟 Highlights

Achieves ~99% accuracy on MNIST test data

Fully modular codebase (data, model, training, evaluation, visualization)

Professional folder structure suitable for academic and industry use

Confusion matrix + prediction samples included

Saved trained model for reuse

Clean and readable notebook for teaching and demonstration

🚀 How to Run

Install dependencies

pip install -r requirements.txt

Train the model

python src/train.py

Evaluate the model

python src/evaluate.py

Generate visualizations

python src/visualize.py

🔮 Future Improvements

Add dropout layers to reduce overfitting

Experiment with deeper CNN architectures

Convert model to TensorFlow Lite for mobile deployment

Add SVM baseline comparison

Deploy model using FastAPI or Flask

Add automated unit tests for each module

📜 License

MIT License

This README is optimized for recruiters, instructors, and portfolio presentation. You can directly upload this file to GitHub.