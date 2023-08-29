import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Define learning rate
    lr_rate = 1e-5
    weight_decay = 1e-6

    # Define batch size
    BATCH_SIZE = 1

    # Image dimensions
    IMG_HEIGHT, IMG_WIDTH = 384, 384

    # Path to directory
    train_dir = r''
    validation_dir = r''
    output_dir = r''

    # Define the transformations
    transform = transforms.Compose([transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load images from directories
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_data = datasets.ImageFolder(root=validation_dir, transform=transform)

    # Create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=5)
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_xl_in21k', pretrained=False, dropout=0.2, stochastic_depth=0.0)

    # Get the number of input features to the final layer
    num_features = model.head.classifier.in_features

   # Create new classifier
    new_classifier = nn.Sequential(
        nn.Linear(num_features, 1920),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(1920, 960),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(960, 960),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(960, 1)
    )

    # Create new head, including the existing layers except the classifier
    new_head = nn.Sequential(
        model.head.bottleneck,
        model.head.avgpool,
        model.head.flatten,
        model.head.dropout,
        new_classifier
    )

    # Replace the head of the model
    model.head = new_head

    print(f'The model has {count_parameters(model)} trainable parameters.')

    # Load model to device
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    # Initialize lists to save loss and accuracy values
    loss_values = []
    val_loss_values = []
    accuracy_values = []
    val_accuracy_values = []

    # Define number of epoch
    n_epochs = 50

    #ema = EMA(model, decay=0.999)
    #ema.register()

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))

        labels_list = []
        for batch in train_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Keep all labels to calculate class weight.
            labels_list.append(labels.detach().cpu().numpy().ravel())

            optimizer.zero_grad()

            outputs = model(inputs).view(-1)
            labels = labels.float()

            # Compute class weights
            class_samples = np.bincount(train_data.targets)
            weights = 1. / torch.tensor(class_samples, dtype=torch.float)
            weights = weights / weights.sum()
            pos_weight = weights[1] / weights[0]
            pos_weight = pos_weight.to(device)

            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            #ema.update()

            total_loss += loss.item()

            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            preds = torch.round(probs)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            tqdm_bar.set_postfix({'Training Loss': '{:.3f}'.format(loss.item()/len(batch)), 'Training Accuracy': '{:.3f}'.format(total_correct/total_samples)})
            tqdm_bar.update()

        loss_values.append(total_loss / len(train_dataloader))
        accuracy_values.append(total_correct / total_samples)

        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pt")

        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        tqdm_bar = tqdm(valid_dataloader, desc=f'Validation Epoch {epoch} ', total=int(len(valid_dataloader)))

        #ema.apply_shadow()

        with torch.no_grad():
            for batch in valid_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).view(-1)
                labels = labels.float()

                # Compute class weights
                class_samples = np.bincount(train_data.targets)
                weights = 1. / torch.tensor(class_samples, dtype=torch.float)
                weights = weights / weights.sum()
                pos_weight = weights[1] / weights[0]
                pos_weight = pos_weight.to(device)

                loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                val_loss = loss_func(outputs, labels)

                total_val_loss += val_loss.item()

                # Convert outputs to probabilities
                probs = torch.sigmoid(outputs)
                preds = torch.round(probs)
                total_val_correct += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

                tqdm_bar.set_postfix({'Validation Loss': '{:.3f}'.format(val_loss.item()/len(batch)), 'Validation Accuracy': '{:.3f}'.format(total_val_correct/total_val_samples)})
                tqdm_bar.update()
            
        #ema.restore()

        val_loss_values.append(total_val_loss / len(valid_dataloader))
        val_accuracy_values.append(total_val_correct / total_val_samples)

        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_values, color='blue', label='Training Loss')
        plt.plot(val_loss_values, color='red', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plot accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_values, color='blue', label='Training Accuracy')
        plt.plot(val_accuracy_values, color='red', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{output_dir}/training_plot_epoch_{epoch+1}.png')
        plt.close()