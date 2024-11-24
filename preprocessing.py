
from dotenv import load_dotenv
import os
import cv2
import albumentations as A

load_dotenv()
TRAIN_DIR = os.getenv('TRAIN_DIR')
AUGMENT = os.getenv('AUGMENT')
MAX = 1336

def augment(file, path, number):
    img = cv2.imread(path)


    ''' Transformations: 
        Geometric: HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomRotate90
        Color: RandomBrightnessContrast, HueSaturationValue, RandomGamma
        Noise and Blur: GaussianBlur, MotionBlur, ISONoise
        Zoom: RandomScale
        Perspective and Occlusions: Perspective, CoarseDropout
    '''
    transform = A.Compose([

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.7),
        A.RandomRotate90(p=0.5),
        
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.RandomGamma(p=0.5),
        
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.ISONoise(p=0.3),
        
      
        A.RandomScale(scale_limit=0.1, p=0.5),
        
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.CoarseDropout(max_holes=5, max_height=50, max_width=50, p=0.3),
    ])

    transformed = transform(image=img)
    transformed_image = transformed['image']

    cv2.imwrite(f'{path[:-4]}_augmented_{number}.jpg', transformed_image)
 
print ("Starting....")
for dir in os.listdir(TRAIN_DIR):
    length = len(os.listdir(f'{TRAIN_DIR}/{dir}'))

    while length < MAX:
        for file in os.listdir(f'{TRAIN_DIR}/{dir}'):

            if length < MAX:
                augment(file, f'{TRAIN_DIR}/{dir}/{file}', length)
                length += 1  




                         
print('Augmented images.')


