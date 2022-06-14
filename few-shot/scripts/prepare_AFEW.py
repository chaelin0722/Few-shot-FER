"""
Run this script to prepare the miniImageNet dataset.

This script uses the 100 classes of 600 images each used in the Matching Networks paper. The exact images used are
given in data/mini_imagenet.txt which is downloaded from the link provided in the paper (https://goo.gl/e3orz6).

1. Download files from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view and place in
    data/miniImageNet/images
2. Run the script
"""
import sys
sys.path.append("../")
from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir


# Clean up folders
rmdir('/home/ivpl-d28/Pycharmprojects/FER/AFEW/trial4_0614_DATA/images_background')
rmdir('/home/ivpl-d28/Pycharmprojects/FER/AFEW/trial4_0614_DATA/images_evaluation')
mkdir('/home/ivpl-d28/Pycharmprojects/FER/AFEW/trial4_0614_DATA/images_background')
mkdir('/home/ivpl-d28/Pycharmprojects/FER/AFEW/trial4_0614_DATA/images_evaluation')


EMOTIONS = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral', 5:'Sad', 6:'Surprise'}
emotions_background = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
emotions_evaluation = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']


for i in range(4):
    mkdir(f'/home/ivpl-d28/Pycharmprojects/FER/AFEW/trial4_0614_DATA/images_background/{emotions_background[i]}')
for i in range(3):
    mkdir(f'/home/ivpl-d28/Pycharmprojects/FER/AFEW/trial4_0614_DATA/images_evaluation/{emotions_evaluation[i]}')



# WHEN THE FOLDERS ARE MIXED WITH NUMBERS, NOT KNWOING EMOTION FOLDER

RAW_DATA_PATH = '/home/ivpl-d28/Pycharmprojects/FER/AFEW/Train_AFEW/'
DATA_PATH = '/home/ivpl-d28/Pycharmprojects/FER/AFEW/Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points/Faces'

dirs = os.listdir(DATA_PATH)


for emotion in EMOTIONS:
    emo_path = os.path.join(RAW_DATA_PATH + EMOTIONS[emotion])
    if EMOTIONS[emotion] in emotions_background:
        category = 'images_background'
    else :
        category = 'images_evaluation'
    for folder in os.listdir(emo_path):
        if folder.split(".")[0] in dirs:
            src = f'{DATA_PATH}/{folder.split(".")[0]}'
            dst = f'/home/ivpl-d28/Pycharmprojects/FER/AFEW/{category}/{EMOTIONS[emotion]}/{folder.split(".")[0]}'
            shutil.copytree(src, dst)
            # shutil.copytree("./test1", "./test2")
            

