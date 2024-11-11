
from dotenv import load_dotenv
import torch
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch import optim
import os
import matplotlib.pyplot as plt
from datetime import datetime
import math

inicio = datetime.now()
print(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

''' Check if GPU is available '''
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
''' Calculate the mean and standard deviation of the dataset '''
def calculates_std_mean(dataloader):
    mean = 0.
    std = 0.
    num_samples = 0.

    for data, _ in dataloader:

        batch_samples = data.size(0)

        ''' Redimension the data to (number of samples, number of channels, number of pixels) '''
        data = data.view(batch_samples, data.size(1), -1)

        ''' Update the mean and standard deviation '''
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)

        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    return mean, std


''' Load the environment variables '''
load_dotenv()
TRAIN_DIR = os.getenv('TRAIN_DIR')
TEST_DIR = os.getenv('TEST_DIR')
VALIDATION_DIR = os.getenv('VALIDATION_DIR')
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))


''' Define the transformations '''
set_transforms = {  
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4713, 0.5089, 0.4957], std = [0.1775, 0.1849, 0.2000]),
        transforms.Resize((224, 224))
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4713, 0.5089, 0.4957], std = [0.1775, 0.1849, 0.2000]),
        transforms.Resize((224, 224))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4713, 0.5089, 0.4957], std = [0.1775, 0.1849, 0.2000]),
        transforms.Resize((224, 224))
    ])
}


''' Create a DataLoader for the images  (classification by family) '''
dataset_train = datasets.ImageFolder(TRAIN_DIR, transform=set_transforms['train'])
dataset_validation = datasets.ImageFolder(VALIDATION_DIR, transform=set_transforms['val'])

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=32, shuffle=True)

''' Calculate the mean and standard deviation of the dataset '''
#mean, std = calculates_std_mean(dataloader) --> RESULTS: Mean: tensor([0.4713, 0.5089, 0.4957]) & Standard deviation: tensor([0.1775, 0.1849, 0.2000])

''' Loading the ResNet model '''
device = get_device()

print("Running on", device)

''' Setting the model '''
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(2048, 41)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

''' Freezes the weights of the model except the last layer '''
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

model.to(device)    

train_loss = []
val_loss = []
best_loss = math.inf
loss_validation = math.inf
current_patience = 0

for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0.0
    true_labels = []
    pred_labels = []

    ''' --- TRAINING THE MODEL --- '''

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True) 

    ''' If epochs higher than 3, unfreeze the weights of the model '''
    if epoch == 3:
        for name, param in model.named_parameters():
            param.requires_grad = True

    with torch.set_grad_enabled(True):
        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
        

        ''' Calculate the loss of the model '''
        epoch_loss = running_loss / len(dataloader_train.dataset)
        train_loss.append(epoch_loss)

        ''' Print the results '''
        print("[Train] Epoch:", epoch, "Loss:", epoch_loss)
            

    ''' --- VALIDATION OF THE MODEL --- '''
    model.eval()
    running_loss = 0.0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in dataloader_validation:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss_validation = criterion(outputs, labels)

            running_loss += loss_validation.item() * images.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

        ''' Calculate the loss of the model '''
        epoch_loss = running_loss / len(dataloader_validation.dataset)
        val_loss.append(epoch_loss)
        
        ''' Print the results '''
        print("[Validation] Epoch:", epoch, "Loss:", epoch_loss)
        
        print("----------------------------------------------")

    ''' Save the model '''
    if loss_validation < best_loss:
        best_loss = loss_validation
        current_patience = 0
        torch.save(model.state_dict(), 'best_model.pth')

    else:
        current_patience += 1

    if current_patience == 3:
        break

    print("----------------------------------------------")
         
''' Plots the training and validation loss '''
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')

fim = datetime.now()
print(f"Término: {fim.strftime('%Y-%m-%d %H:%M:%S')}")

duracao = fim - inicio
print(f"Duração total: {duracao}")
print(f"Best loss: {best_loss}")