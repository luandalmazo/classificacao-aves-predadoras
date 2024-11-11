
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
MAIN_DIR = os.getenv('MAIN_DIR')
TRAIN_DIR = os.getenv('TRAIN_DIR')
TEST_DIR = os.getenv('TEST_DIR')
VALIDATION_DIR = os.getenv('VALIDATION_DIR')
PERCENTAGE_TEST = 0.30
PERCENTAGE_VALIDATION = 0.25
TOTAL_TEST = 0
TOTAL_TRAIN = 0
TOTAL_VALIDATION = 0

length = 0
total = 0

''' Moving files to the test and train directories '''
for entry in os.listdir(MAIN_DIR):
        ''' Creates entry inside TRAIN_DIR, TEST_DIR and VALIDATION_DIR '''
        if not entry.endswith('.gz'):
            if not os.path.exists(f'{TRAIN_DIR}/{entry}'):
                os.makedirs(f'{TRAIN_DIR}/{entry}')
            if not os.path.exists(f'{TEST_DIR}/{entry}'):
                os.makedirs(f'{TEST_DIR}/{entry}')
            if not os.path.exists(f'{VALIDATION_DIR}/{entry}'):
                os.makedirs(f'{VALIDATION_DIR}/{entry}')

                
            if os.path.isdir(f'{MAIN_DIR}/{entry}'):
                for subdirs in os.listdir(f'{MAIN_DIR}/{entry}'):
                    if not os.path.exists(f'{TRAIN_DIR}/{entry}/{subdirs}'):
                        os.makedirs(f'{TRAIN_DIR}/{entry}/{subdirs}')
                    if not os.path.exists(f'{TEST_DIR}/{entry}/{subdirs}'):
                        os.makedirs(f'{TEST_DIR}/{entry}/{subdirs}')
                    if not os.path.exists(f'{VALIDATION_DIR}/{entry}/{subdirs}'):
                        os.makedirs(f'{VALIDATION_DIR}/{entry}/{subdirs}')
                    
                    length = 0
                    total = PERCENTAGE_TEST * len(os.listdir(f'{MAIN_DIR}/{entry}/{subdirs}'))
                    for files in os.listdir(f'{MAIN_DIR}/{entry}/{subdirs}'):
                        
                        if length < total:
                            length += 1

                            ''' Moves the file to the test directory '''
                            shutil.move(f'{MAIN_DIR}/{entry}/{subdirs}/{files}', f'{TEST_DIR}/{entry}/{subdirs}/{files}')
                        else:
                            ''' Moves the file to the train directory '''
                            shutil.move(f'{MAIN_DIR}/{entry}/{subdirs}/{files}', f'{TRAIN_DIR}/{entry}/{subdirs}/{files}')

print('Moved files to the test and train directories.')


for entry in os.listdir(TRAIN_DIR):
    if os.path.isdir(f'{TRAIN_DIR}/{entry}'):
        for subdirs in os.listdir(f'{TRAIN_DIR}/{entry}'):

            length = 0
            total = PERCENTAGE_VALIDATION * len(os.listdir(f'{TRAIN_DIR}/{entry}/{subdirs}'))
            for files in os.listdir(f'{TRAIN_DIR}/{entry}/{subdirs}'):

                if (length < total):
                    length += 1
                    
                    ''' Moves the file to the validation directory '''
                    shutil.move(f'{TRAIN_DIR}/{entry}/{subdirs}/{files}', f'{VALIDATION_DIR}/{entry}/{subdirs}/{files}')

print('Moved files to the validation directory.')
