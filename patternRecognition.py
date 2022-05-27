from tabnanny import verbose
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from setup import *
from utils import classification_pipeline
import os
import math

def preprocessor (xarray):
    '''
    When using ImageDataGenerator for grayscale image, we must use this to convert RGB to Grayscale
    '''
    xarray=(xarray/127.5)-1
    return xarray
preprocessing_function =preprocessor
'''
Using ImageDatagenerator to load dataset 
Params:
    - batch_size: 
    - class_mode: use "binary" to classify 2 classes, and use "categorical" to classify multiclasses
    - seed: create random with stable results all tries, not changable when running
    - color_mode: default to rgb, or grayscale
    - target_size: your image size
'''
train_datagen = ImageDataGenerator(
  preprocessing_function=preprocessing_function)

train_generator = train_datagen.flow_from_directory(
  SAVE_PATH + 'train',
  batch_size=BATCH_SIZE,
  class_mode='categorical',
  seed=42,
  color_mode='grayscale',
  target_size=(IMG_HEIGHT,IMG_WIDTH))

test_datagen = ImageDataGenerator(
  preprocessing_function=preprocessing_function)

test_generator = test_datagen.flow_from_directory(
  SAVE_PATH + 'test',
  shuffle=False,
  class_mode='categorical',
  seed=42,
  color_mode='grayscale',
  target_size=(IMG_HEIGHT,IMG_WIDTH))
# Modify dataset
x_train = []
y_train = []
x_test = []
y_test = []
counter = 0
step_size_train = math.floor(train_generator.n/train_generator.batch_size)
step_size_val =  math.floor(test_generator.samples // test_generator.batch_size)

for x,y in train_generator:
  x_train.append(x)
  y_train.append(y)
  counter = counter + 1
  if counter == step_size_train:
    break
counter = 0
for x,y in test_generator:
  x_test.append(x)
  y_test.append(y)
  counter = counter + 1
  if counter == step_size_val:
    break
x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1],) + x_train.shape[2:])

y_train = np.array(y_train)
y_train = np.reshape(y_train, (y_train.shape[0]*y_train.shape[1],) + y_train.shape[2:])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0]*x_test.shape[1],) + x_test.shape[2:])

y_test = np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1],) + y_test.shape[2:])

x_train=x_train.reshape(x_train.shape[0],IMG_HEIGHT,IMG_WIDTH,1)
x_test=x_test.reshape(x_test.shape[0],IMG_HEIGHT,IMG_WIDTH,1)

# Load model
filepath="checkpoint/22-04-2022.hdf5"
model = load_model(filepath)
# Extractor
new_model=Model(inputs=model.input,outputs=model.layers[-2].output)
train_x=new_model.predict(x_train)
test_x=new_model.predict(x_test)
train_y= np.asarray(tf.argmax(y_train, axis=1)).ravel()
train_x=train_x.reshape(-1,128)
test_x=test_x.reshape(-1,128)

##################################################### Classìication algorithm ####################################
'''
*** KNN Algorithm ***
'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
k_range = list(range(1, 20))
param_grid_knn = dict(n_neighbors=k_range)
model, pred = classification_pipeline(train_x, test_x, train_y, test_generator, knn, 
                                 param_grid_knn, cv=5, scoring_fit='accuracy', title='KNN')


'''
*** SVM Algorithm ***
'''                                
from sklearn.svm import SVC
param_grid_svm = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
svc = SVC()
model, pred = classification_pipeline(train_x, test_x, train_y, test_generator, svc, 
                                 param_grid_svm, cv=5, scoring_fit='accuracy', title='SVM')


'''
*** Decision Tree Algorithm ***
'''                                
from sklearn.tree import DecisionTreeClassifier
param_grid_dtree = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
dtree_model=DecisionTreeClassifier()
model, pred = classification_pipeline(train_x, test_x, train_y, test_generator, dtree_model, 
                                 param_grid_dtree, cv=5, scoring_fit='accuracy', title='Decision Tree')


'''
*** Random Forest Algorithm ***
'''   
from sklearn.ensemble import RandomForestClassifier
param_grid_rf = {
    'n_estimators': [400, 700, 1000],
    'max_depth': [15,20,25],
    'max_leaf_nodes': [50, 100, 200]
}
rfc=RandomForestClassifier()
model, pred = classification_pipeline(train_x, test_x, train_y, test_generator, rfc, 
                                 param_grid_rf, cv=5, scoring_fit='accuracy', title='Random Forest')


'''
*** XGBoost Algorithm ***
'''                               
# import xgboost
# xgb = xgboost.XGBClassifier()         
# param_grid = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],                                        
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#         }     
# model, pred = classification_pipeline(train_x, test_x, train_y, test_generator, xgb, 
#                                  param_grid, cv=5, scoring_fit='accuracy')


'''
*** Naive Bayes Algorithm ***
'''  
from sklearn.naive_bayes import GaussianNB
param_grid_gnb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
gnb = GaussianNB()
model, pred = classification_pipeline(train_x, test_x, train_y, test_generator, gnb, 
                                 param_grid_gnb, cv=5, scoring_fit='accuracy', title='Naive Bayes')
