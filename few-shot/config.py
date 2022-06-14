import os


PATH = os.path.dirname(os.path.realpath(__file__))

#DATA_PATH = '/home/ivpl-d28/Pycharmprojects/FER/AFEW/Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points/Faces'
DATA_PATH = '/home/ivpl-d28/Pycharmprojects/FER/AFEW'
#/home/ivpl-d28/Pycharmprojects/FER/few-shot/data' #None

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
