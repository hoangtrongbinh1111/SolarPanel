import json
import os
import shutil
import random
import math
from setup import *

'''
Init dict to save list image and label
'''
json_file = open('dataset/module_metadata.json')
module_metadata = json.load(json_file)
imgArr = {
  "Cell":[],
  "Cell-Multi":[],
  "Cracking":[],
  "Diode":[],
  "Diode-Multi":[],
  "Hot-Spot":[],
  "Hot-Spot-Multi":[],
  "No-Anomaly":[],
  "Offline-Module":[],
  "Shadowing":[],
  "Soiling":[],
  "Vegetation":[]
}
for index in module_metadata:
  imgArr[module_metadata[index]["anomaly_class"]].append(module_metadata[index]["image_filepath"])

def initSplitFolder():
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    listType = ["train", "val", "test"]
    listLabel = ["Cell","Cell-Multi","Cracking","Diode","Diode-Multi","Hot-Spot","Hot-Spot-Multi","No-Anomaly","Offline-Module","Shadowing","Soiling","Vegetation"]
    for typ in listType:
        splitFold = SAVE_PATH + typ + '/'
        if not os.path.exists(splitFold):
            os.makedirs(splitFold)
        for lab in listLabel:
            splitFile = splitFold + lab
            if not os.path.exists(splitFile):
                os.makedirs(splitFile)

def divideDataset(dataset,type):
  for label, img_path in dataset.items():
    if type == 'train':
      lenImg = math.floor(len(img_path)*train_ratio)
      list_image_split = img_path[0:lenImg]
    elif type == 'val':
      lenImgStart = math.floor(len(img_path)*train_ratio)
      lenImgEnd = math.floor(len(img_path)*val_ratio) + lenImgStart
      list_image_split = img_path[lenImgStart:lenImgEnd]
    elif type == 'test':
      lenImgStart = math.floor(len(img_path)*train_ratio) + math.floor(len(img_path)*val_ratio)
      lenImgEnd = len(img_path) + 1
      list_image_split = img_path[lenImgStart:lenImgEnd]
    for img in list_image_split:
      dirClass = SAVE_PATH + type + '/' + label
      train_folder = dirClass
      train_file = ROOT_PATH + img
      dest_file = img.split('/')[1] # split images/....jpg to ...jpg
      dest_file = train_folder + '/' + dest_file
      shutil.copyfile(train_file, dest_file)
  print("Done split for - "+type)

initSplitFolder()
divideDataset(imgArr,'train')
divideDataset(imgArr,'val')
divideDataset(imgArr,'test')