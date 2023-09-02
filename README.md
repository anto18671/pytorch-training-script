# [PyTorch Training Script with EfficientNetV2](https://github.com/hankyul2/EfficientNetV2-pytorch)

This script offers a straightforward implementation for training an image classifier using **PyTorch** and **EfficientNetV2**.

## 📋 Table of Contents
- [Dependencies](#-dependencies)
- [Script Structure](#-script-structure)
- [Highlights](#-highlights)
- [How to Run](#-how-to-run)
- [Note](#⚠️-note)

## 🛠 Dependencies
- PyTorch
- torchvision
- tqdm
- numpy
- matplotlib

## 📖 Script Structure

### 1. **Importing Libraries**
All the essential packages for data processing, model definition, training, and visualization are imported.

### 2. **EMA (Exponential Moving Average) Class**
A utility class to assist in maintaining an exponential moving average of model parameters. This is particularly beneficial for validation.

### 3. **Utility Function**
- `count_parameters`: A function to return the count of trainable parameters in a given model.

### 4. **Main Script**
This script encompasses the initialization of hyperparameters, data transformations, dataset loading, model structure definition, and execution of the training loop. After each epoch concludes, the model's weights are saved, and both the training and validation losses and accuracies are plotted.

## 🌟 Highlights

- **Data Loading and Transformation:** Employs `ImageFolder` to load images from directories and subsequently applies a series of transformations for data preprocessing.

- **Model:** Leverages the EfficientNetV2 architecture available in `torch.hub`. The classifier head of the pre-trained model is replaced with a custom classifier tailored for binary classification.

- **Training Loop:** The loop makes use of the AdamW optimizer combined with the Binary Cross Entropy with Logits loss. Additionally, class weights are considered to counteract any class imbalances.

- **EMA (Exponential Moving Average) Integration:** For smoother validation, you can enable the EMA during training. Uncomment the EMA initialization and its corresponding methods in the training loop to utilize this feature.

- **Learning Rate Finder:** The script employs an adaptive learning rate approach, ensuring optimal convergence speed. Users can adjust the learning rate and monitor its effect on training loss for better performance.

- **Visualization:** Post each epoch, the script visualizes and saves the training and validation loss and accuracy as PNG images.

## 🚀 How to Run

1. **Set Paths:**
```
train_dir = r'path_to_training_data'
validation_dir = r'path_to_validation_data'
output_dir = r'path_to_output_directory'
```


2. Run the script:
```
python train.py
```

3. The trained models will be saved in the `output_dir` as 'model_{epoch_number}.pt'. Additionally, training plots for each epoch will also be saved in the same directory.

⚠️ Note
Before running the script, ensure you have adequate storage and computational resources, especially if you're training on high-resolution images with a deep neural network like EfficientNetV2.