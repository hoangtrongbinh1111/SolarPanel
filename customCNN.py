from select import epoll
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, add
from keras.utils.vis_utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from setup import *
import numpy as np
from PIL import Image
import os
##################################### CONFIG GPUS ########################################
from tensorflow.python.client import device_lib 
##################################### LOAD DATA ###########################################
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
  horizontal_flip=True,
  vertical_flip=True,
  zoom_range=0.2,
  fill_mode='nearest',
  preprocessing_function=preprocessing_function)

train_generator = train_datagen.flow_from_directory(
  SAVE_PATH + 'train',
  batch_size=BATCH_SIZE,
  class_mode='categorical',
  seed=42,
  color_mode='grayscale',
  target_size=(IMG_HEIGHT,IMG_WIDTH))

validation_datagen = ImageDataGenerator(
  preprocessing_function=preprocessing_function)

validation_generator = validation_datagen.flow_from_directory(
  SAVE_PATH + 'val',
  shuffle=False,
  class_mode='categorical',
  seed=42,
  color_mode='grayscale',
  target_size=(IMG_HEIGHT,IMG_WIDTH))

##################################### BUILD OUR CUSTOM MODEL ###########################################
'''
Create residual block with 2 Convnet
'''
def residual_module(layer_in, n_filters, strides):
  merge_input = layer_in
  # conv1
  conv1 = Conv2D(n_filters, (3,3),strides=strides, padding='same', activation='relu',kernel_regularizer=l2(0.01),
  kernel_initializer='he_normal')(layer_in)
  conv1 = BatchNormalization()(conv1)
  # conv2
  conv2 = Conv2D(n_filters, (3,3),strides=1, padding='same', activation='relu',kernel_regularizer=l2(0.01),
  kernel_initializer='he_normal')(conv1)
  conv2 = BatchNormalization()(conv2)
  if strides == 2:
    merge_input = Conv2D(n_filters, (1,1),strides=strides, padding='same', activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(0.01),)(merge_input)
  # add filters, assumes filters/channels last
  layer_out = add([conv2, merge_input])
  return layer_out

'''
Build custom Model base on backbone Resnet
'''
input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)) # 1 is your channel
layer = Conv2D(filter, (3, 3), strides=(2,1),
                              kernel_initializer="he_normal",kernel_regularizer=l2(0.01),)(input)
layer = residual_module(layer, filter, 2)
layer = residual_module(layer, filter, 1)
layer = residual_module(layer, filter, 1)
layer = residual_module(layer, filter*2, 2)
layer = residual_module(layer, filter*2, 1)
layer = residual_module(layer, filter*2, 1)
layer = GlobalAveragePooling2D()(layer)
fc = Dense(num_classes,kernel_regularizer=l2(0.01), activation="softmax")(layer)
model = Model(inputs=input, outputs=fc)
# summarize model
model.summary()

##################################### COMPILE MODEL AND TRAINING###########################################
'''
Compile your model
'''
optimizer = tf.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

'''
Checkpoint to save your training process
'''
checkpoint = ModelCheckpoint(checkpoint_best, monitor='val_loss', verbose=1, save_weights_only = False, save_best_only=True, mode='min')
tfboard = TensorBoard(log_dir=logdir)
earlyStopping = EarlyStopping(monitor='val_loss', patience=patience_to_stop)
callbacks_list = [checkpoint, tfboard, earlyStopping]

'''
Process training set
'''
step_size_train = train_generator.n/train_generator.batch_size
step_size_val = validation_generator.samples // validation_generator.batch_size
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator, 
                   validation_steps =step_size_val,
                   callbacks = callbacks_list,
                   epochs = epochs)
np.save(history_best,history.history)