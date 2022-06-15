import numpy as np
import cv2
import os
import math
from tqdm import tqdm
from skimage import transform as trans
from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)

#DATA_DIR = "/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/TRAIN/AlignedFaces_LBPTOP_Points/"
DATA_DIR = "/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/VALID/"

def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

#print(get_iou([10,10,20,20],[15,15,25,25]))

def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = [224,224]
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==224:
        src[:,0] += 8.0
    src*=2
    if landmark is not None:
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        #dst=dst[:3]
        #src=src[:3]
        #print(dst.shape,src.shape,dst,src)
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        #print(M)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
              det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin//2, 0)
        bb[1] = np.maximum(det[1]-margin//2, 0)
        bb[2] = np.minimum(det[2]+margin//2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin//2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
              ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else: #do align using landmark
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped

def save_aligned_faces(source_path,save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for emotion_folder in tqdm(os.listdir(source_path)):
        if not os.path.exists(os.path.join(save_path, emotion_folder)):
            emotion_path = os.path.join(save_path, emotion_folder)
            os.mkdir(emotion_path)
            # /home/ivpl
            prev_b = None
            counter = 0
            for image_folder in os.listdir(os.path.join(source_path, emotion_folder)):
                if not os.path.exists(os.path.join(emotion_path, image_folder)):
                    os.mkdir(os.path.join(emotion_path, image_folder))

                    for image in sorted(os.listdir(os.path.join(source_path, emotion_folder, image_folder))):
                        filename = os.path.join(source_path, emotion_folder, image_folder, image)
                        frame = cv2.imread(filename)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        bounding_boxes, points = imgProcessing.detect_faces(frame)
                        points = points.T

                        best_ind=None
                        if len(bounding_boxes)==0:
                            print('No faces found for ',filename)
                            counter+=1
                            if prev_b is None or counter>3:
                                continue
                            else:
                                b=prev_b
                        elif len(bounding_boxes)>1:
                            print('Too many faces (',len(bounding_boxes),') found for ',filename)
                            if prev_b is None:
                                #continue
                                best_ind=0
                                b=[int(bi) for bi in bounding_boxes[best_ind]]
                                counter=0
                            else:
                                best_iou=0
                                for i in range(len(bounding_boxes)):
                                    iou=get_iou(bounding_boxes[i],prev_b)
                                    if iou>best_iou:
                                        best_iou=iou
                                        best_ind=i
                                if best_iou>0:
                                    b=[int(bi) for bi in bounding_boxes[best_ind]]
                                    print('best_iou (',best_iou,') best_bb ',bounding_boxes[best_ind])
                                else:
                                    #continue
                                    best_ind=0
                                    b=[int(bi) for bi in bounding_boxes[best_ind]]
                                    counter=0
                        else:
                            best_ind=0
                            b=[int(bi) for bi in bounding_boxes[best_ind]]
                            counter=0
                        prev_b=b

                        if True:
                            p=None
                            if best_ind is not None:
                                p=points[best_ind]
                                if True: #not USE_RETINA_FACE:
                                    p = p.reshape((2,5)).T
                            face_img=preprocess(frame,b,p) #None, #p)
                        else:
                            x1,y1,x2,y2=b[0:4]
                            face_img=frame[y1:y2,x1:x2,:]
                        if np.prod(face_img.shape)==0:
                            print('Empty face ',b,' found for ',filename)
                            continue

                        cv2.imwrite(os.path.join(save_path, emotion_folder, image_folder, image), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

#aligned
save_aligned_faces(os.path.join(DATA_DIR, 'color_Faces/'), os.path.join(DATA_DIR, 'aligned/'))
