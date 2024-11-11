import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns 

set_transforms = {  
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4713, 0.5089, 0.4957], std = [0.1775, 0.1849, 0.2000]),
        transforms.Resize((224, 224))
    ])
}

load_dotenv()
TEST_DIR = os.getenv('TEST_DIR')
dataset_test = datasets.ImageFolder(TEST_DIR, transform=set_transforms['test'])
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

''' Check if GPU is available '''
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

''' Testing the model '''
print("Starting...")
model = models.resnet50()
model.fc = nn.Linear(2048, 41)
model.load_state_dict(torch.load("./best_model.pth"))
correct = 0
total = 0
device = get_device()
model.to(device)
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for data in dataloader_test:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())


precision = precision_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
precision_class = precision_score(true_labels, predicted_labels, average=None)
f1_class = f1_score(true_labels, predicted_labels, average=None)
recall_class = recall_score(true_labels, predicted_labels, average=None)

print(f"Precisão: {precision}")
print(f"F1: {f1}")
print(f"Recall: {recall}")
print("Precisão por classe:")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {precision_class[idx]}")
print("------------------------")
print("F1 por classe:")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {f1_class[idx]}")
print("------------------------")
print("Recall por classe:")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {recall_class[idx]}")

cm = confusion_matrix(true_labels, predicted_labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.set(rc={'figure.figsize':(36,12)})
sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.2f')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('./confusion_matrix_new.png')
