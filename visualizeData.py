import os
from setup import *
'''
Statistical number of images per classes
'''
def statImage():
    for typ in listType:
        splitFold = SAVE_PATH + typ + '/'
        print(f'=============={typ}===============')
        for lab in listLabel:
            splitFile = splitFold + lab
            count = 0
            for img in os.listdir(splitFile):
                count += 1
            print(f'Number of images in class "{lab}" ===> {count}')

statImage()