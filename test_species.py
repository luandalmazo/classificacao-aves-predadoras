import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, f1_score, recall_score
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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def test_model(model, dataloader_test, device):
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
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    precision = precision_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision_class = precision_score(true_labels, predicted_labels, average=None)
    f1_class = f1_score(true_labels, predicted_labels, average=None)
    recall_class = recall_score(true_labels, predicted_labels, average=None)

    return precision, f1, recall, precision_class, f1_class, recall_class

device = get_device()
model_paths = ["./best_model_1.pth", "./best_model_2.pth", "./best_model_3.pth", "./best_model_4.pth", "./best_model_5.pth"]

all_precision = []
all_f1 = []
all_recall = []
all_precision_class = []
all_f1_class = []
all_recall_class = []

for model_path in model_paths:
    model = models.resnet50()
    model.fc = nn.Linear(2048, 41)
    model.load_state_dict(torch.load(model_path))
    
    precision, f1, recall, precision_class, f1_class, recall_class = test_model(model, dataloader_test, device)
    
    all_precision.append(precision)
    all_f1.append(f1)
    all_recall.append(recall)
    all_precision_class.append(precision_class)
    all_f1_class.append(f1_class)
    all_recall_class.append(recall_class)

mean_precision = np.mean(all_precision)
std_precision = np.std(all_precision)

mean_f1 = np.mean(all_f1)
std_f1 = np.std(all_f1)

mean_recall = np.mean(all_recall)
std_recall = np.std(all_recall)

mean_precision_class = np.mean(all_precision_class, axis=0)
std_precision_class = np.std(all_precision_class, axis=0)

mean_f1_class = np.mean(all_f1_class, axis=0)
std_f1_class = np.std(all_f1_class, axis=0)

mean_recall_class = np.mean(all_recall_class, axis=0)
std_recall_class = np.std(all_recall_class, axis=0)

print("------------------------")
print(f"Precisão média: {mean_precision:.2f} ± {std_precision:.2f}")
print(f"F1 média: {mean_f1:.2f} ± {std_f1:.2f}")
print(f"Recall médio: {mean_recall:.2f} ± {std_recall:.2f}")

print("\nPrecisão média por classe:")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {mean_precision_class[idx]:.2f} ± {std_precision_class[idx]:.2f}")
print("------------------------")

print("F1 médio por classe:")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {mean_f1_class[idx]:.2f} ± {std_f1_class[idx]:.2f}")
print("------------------------")

print("Recall médio por classe:")
for idx, class_name in enumerate(dataset_test.classes):
    print(f"{class_name}: {mean_recall_class[idx]:.2f} ± {std_recall_class[idx]:.2f}")


