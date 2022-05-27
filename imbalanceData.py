import os
import cv2
import random
import numpy as np
from setup import *
from Oversampling import *

def createNewImageAug(splitFile, filename):
    img = cv2.imread(splitFile+'/'+filename)
    width, height = img.shape[:2]
    width_trans = random.uniform(-width*0.2,width*0.2)
    height_trans = random.uniform(-height*0.2,height*0.2)
    #vertical flip
    img_flip_ud = cv2.flip(img, 0)
    cv2.imwrite(splitFile+'/ver_'+filename, img_flip_ud)
    #horizontal flip
    img_flip_ud = cv2.flip(img, 1)
    cv2.imwrite(splitFile+'/hor_'+filename, img_flip_ud)
    # hor and ver flip
    img_flip_ud = cv2.flip(img, -1)
    cv2.imwrite(splitFile+'/hor_ver_'+filename, img_flip_ud)
    #translation
    translation_matrix = np.float32([ [1,0,height_trans], [0,1,width_trans] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (height, width), cv2.INTER_LINEAR)
    cv2.imwrite(splitFile+'/trans_'+filename, img_translation)
def augmentImage():
    for typ in ["train"]:
        splitFold = SAVE_PATH + typ + '/'
        print(f'=============={typ}===============')
        for lab in listLabel:
            splitFile = splitFold + lab
            if lab != 'No-Anomaly':
                for filename in os.listdir(splitFile):
                    createNewImageAug(splitFile, filename)
                print("Done ===> "+lab)
augmentImage()
# oversampler = Oversampler(SAVE_PATH+'train', 'dataset/test')
# oversample_dict = {'Diode': 0.25, 'Diode-Multi': 0.25, 'Hot-Spot-Multi': 0.25, 'Soiling': 0.25}
# oversampled_df = oversampler.df_val_train_by_pct(valid_pct=0.2, cats_to_pct=oversample_dict)
# oversampler.copy_to_output_with_csv(oversample=True)