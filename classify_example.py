from torchvision import models
from PIL import Image
from torchvision import datasets, transforms
from torch import nn
import torch
import os
from dotenv import load_dotenv

load_dotenv()
TEST_DIR = os.getenv('TEST_DIR')

set_transforms = {  
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4713, 0.5089, 0.4957], std = [0.1775, 0.1849, 0.2000]),
        transforms.Resize((224, 224))
    ])
}

dataset_test = datasets.ImageFolder(TEST_DIR, transform=set_transforms['test'])

''' Open the image '''
image = Image.open("INSERT THE PATH TO THE IMAGE.jpg")

''' Define the transformations '''
set_transforms = {  
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4713, 0.5089, 0.4957], std = [0.1775, 0.1849, 0.2000]),
        transforms.Resize((224, 224))
    ])
}

''' Apply the transformations '''
image = set_transforms['test'](image)
image = image.unsqueeze(0)

''' Load the model '''
model = models.resnet50()

''' Change 41 to 6 if you are using the family model '''
model.fc = nn.Linear(2048, 41) 
model.load_state_dict(torch.load("./best_model.pth"))

''' Make a prediction '''
model.eval()
output = model(image)

''' Normalize the output '''
output = torch.nn.functional.softmax(output, dim=1)

''' print the probabilities for each class and show the classes '''
for idx, prob in enumerate(output[0]):
    print(f"{dataset_test.classes[idx]}: {prob.item()}")

''' Get the label from the prediction '''
prediction = torch.argmax(output)
print(f"The prediction is: {dataset_test.classes[prediction]}")
