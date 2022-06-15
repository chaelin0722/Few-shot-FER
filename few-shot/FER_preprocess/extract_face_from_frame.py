import os
from tqdm import tqdm
import cv2
import numpy as np

from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)

INPUT_SIZE = (224, 224)
#DATA_DIR = "/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/TRAIN/AlignedFaces_LBPTOP_Points/"
DATA_DIR = "/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/VALID/"
#INPUT_PATH = "/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/TRAIN/AlignedFaces_LBPTOP_Points/frames/"
#OUTPUT_PATH = "/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/TRAIN/AlignedFaces_LBPTOP_Points/color_Faces/"

def save_faces(source_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for emotion_folder in tqdm(os.listdir(source_path)):
        if not os.path.exists(os.path.join(save_path, emotion_folder)):
            emotion_path = os.path.join(save_path, emotion_folder)
            os.mkdir(emotion_path)
#/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/TRAIN/AlignedFaces_LBPTOP_Points/frames/Disgust/
            for image_folder in os.listdir(os.path.join(source_path, emotion_folder)):
                if not os.path.exists(os.path.join(emotion_path, image_folder)):
                    os.mkdir(os.path.join(emotion_path, image_folder))

                    for image in os.listdir(os.path.join(source_path, emotion_folder, image_folder)):
                        filename = os.path.join(source_path, emotion_folder, image_folder, image)
                        frame_bgr = cv2.imread(filename)
                        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        bounding_boxes, _ = imgProcessing.detect_faces(frame)

                        if len(bounding_boxes) == 0:
                            print('No faces found for ', filename)
                            face_img = frame_bgr
                            faceFound = 'noface'
                        else:
                            if len(bounding_boxes) > 1:
                                print('Too many faces (', len(bounding_boxes), ') found for ', filename)
                                bounding_boxes = bounding_boxes[:1]

                            b = [int(bi) for bi in bounding_boxes[0]]
                            x1, y1, x2, y2 = b[0:4]
                            face_img = frame_bgr[y1:y2, x1:x2, :]

                            if np.prod(face_img.shape) == 0:
                                print('Empty face ', b, ' found for ', filename)
                                continue

                            faceFound = ''

                            # face_img=cv2.resize(face_img,INPUT_SIZE)
                            root, ext = os.path.splitext(image)
                            cv2.imwrite(os.path.join(save_path, emotion_folder, image_folder, root + faceFound + ext), face_img)
        else:
            print(emotion_folder)


# source_path = "/home/kdemochkin/emotions/train/AlignedFaces_LBPTOP_Points/frames/"
# save_path = "/home/kdemochkin/emotions/train/AlignedFaces_LBPTOP_Points/frames_mtcnn/"
save_faces(os.path.join(DATA_DIR, 'frames/'), os.path.join(DATA_DIR, 'color_Faces'))

