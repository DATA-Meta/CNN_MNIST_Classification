# CNN for MNIST Digit Classification

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. It demonstrates how convolutional layers learn spatial patterns and why CNNs outperform traditional ML models like SVM for image data.

## Objectives
- Build and train a CNN on MNIST
- Experiment with convolution + pooling layers
- Understand feature extraction and spatial hierarchies
- Compare CNN performance with classical ML methods
- Visualize predictions and confusion matrix

## Repository Structure
- notebooks/ – Jupyter notebook with full workflow  
- src/ – Modular Python scripts (data, model, training, evaluation)  
- models/ – Saved trained model  
- results/ – Confusion matrix, training curves, predictions  
- reports/ – Academic PDF report  
- requirements.txt – Dependencies  

## Model Performance
- Training Accuracy: ~99.4%  
- Validation Accuracy: ~99.1%  
- Test Accuracy: ~99.06%  

## How to Run
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/visualize.py


## License
MIT License

