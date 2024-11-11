
from dotenv import load_dotenv
import os
import cv2
import albumentations as A

load_dotenv()
TRAIN_DIR = os.getenv('TRAIN_DIR')
AUGMENT = os.getenv('AUGMENT')

augmentation_factors = {
    "corujinha-do-sul (Megascops sanctaecatarinae)": 0.2, #668
    "corujinha-sapo (Megascops atricapilla)": 0.2, #327
    "gavião-real (Harpia harpyja)": 0.2, #500
    "mocho-dos-banhados (Asio flammeus)": 0.2, #598
    "águia-cinzenta (Urubitinga coronata)": 0.2, #577
    "condor-dos-andes (Vultur gryphus)": 0.3 #111
}

def augment_file_once(file, path):
    img = cv2.imread(path)
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
        A.RandomRotate90(p=1)
    ])

    transformed = transform(image=img)
    transformed_image = transformed['image']

    cv2.imwrite(f'{path[:-4]}_augmented_once.jpg', transformed_image)
 
def augment_file_twice(file, path):
    img = cv2.imread(path)
    transform = A.Compose([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
        A.RandomRotate90(p=1)
    ])

    transformed = transform(image=img)
    transformed_image = transformed['image']
    
    cv2.imwrite(f'{path[:-4]}_augmented_twice.jpg', transformed_image)


if AUGMENT == 'True':
    for keys in augmentation_factors:
        for f in os.listdir(f'{TRAIN_DIR}/{keys}'):
            if not f.endswith('augmented_once.jpg') and not f.endswith('augmented_twice.jpg'):

                if augmentation_factors[keys] == 0.2:
                    path = f'{TRAIN_DIR}/{keys}/{f}'
                    augment_file_once(f, path)

                if augmentation_factors[keys] == 0.3:
                    path = f'{TRAIN_DIR}/{keys}/{f}'
                    for i in range(2):
                        if i == 0:
                            augment_file_once(f, path)
                        else:
                            augment_file_twice(f, path)
                         
print('Augmented images.')


