# pytorch-training-script

This script provides a straightforward implementation for training an image classifier using PyTorch and EfficientNetV2. Below is a brief description of the main components of the script.

Dependencies:
-------------
- PyTorch
- torchvision
- tqdm
- numpy
- matplotlib

Script Structure:
-----------------

1. **Importing Necessary Libraries:**
   All the necessary packages for data processing, model definition, training, and visualization are imported.

2. **EMA (Exponential Moving Average) Class:**
   This utility class helps in maintaining an exponential moving average of model parameters, which can sometimes be useful for validation.

3. **Utility Function:**
   - `count_parameters`: Returns the number of trainable parameters in a given model.

4. **Main Script:**
   The main script initializes hyperparameters, sets up data transformations, loads the dataset, defines the model structure, and carries out the training loop. At the end of each epoch, it saves the model's weights and plots the training and validation loss and accuracy.

Highlights:
-----------

- **Data Loading and Transformation:**
  Uses `ImageFolder` for loading images from directories and applies a series of transformations to preprocess the data. 

- **Model:**
  Utilizes the EfficientNetV2 architecture from the `torch.hub`. Additionally, the script replaces the classifier head of the pre-trained model with a custom-defined classifier suitable for binary classification.

- **Training Loop:**
  Uses a combination of the AdamW optimizer and the Binary Cross Entropy with Logits loss. The script also takes into consideration class weights to handle any class imbalance.

- **Visualization:**
  At the end of each epoch, the script plots the training and validation loss and accuracy, saving the plots as PNG images.

How to Run:
-----------

1. Define the paths:
train_dir = r'path_to_training_data'
validation_dir = r'path_to_validation_data'
output_dir = r'path_to_output_directory'

2. Run the script:
python train.py

3. The trained models will be saved in the `output_dir` as 'model_{epoch_number}.pt'. Additionally, training plots for each epoch will also be saved in the same directory.

Note:
-----
Before running the script, ensure you have adequate storage and computational resources, especially if you're training on high-resolution images with a deep neural network like EfficientNetV2.