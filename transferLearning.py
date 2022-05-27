from calendar import EPOCH
import tensorflow as tf
import os
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

class ModelTrain:

    def __init__(self, model, target_dim, batch_size):
        self.model = model
        self.target_dim = target_dim
        self.batch_size = batch_size
     
    def freeze_layers(self, no_layers):
        
        for layers in self.model.layers[:-no_layers]:
            layers.trainable = True

        for layers in self.model.layers[-no_layers:]:
            layers.trainable = True 
        
    def train(self, train_generator, validation_generator, callbacks = [], epochs = 50):
        
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=callbacks,
            epochs=epochs
        )


class TrainingCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs['acc'] > 0.90:
            print("Reached target training accuracy")
            self.model.stop_training = True

def add_regularization(model, layers, regularizer=tf.keras.regularizers.l2(0.001)):
    for layer in layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
    
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

def get_data_generators(datagen, target_dim = 224, batch_size = 32):
    
    train_generator = datagen.flow_from_directory(
        directory='dataset/dataset12ClassKeras/train/', 
        target_size=(target_dim, target_dim),
        batch_size=batch_size,
    )

    validation_generator = datagen.flow_from_directory(
        directory='dataset/dataset12ClassKeras/val/',
        target_size=(target_dim, target_dim),
        batch_size=batch_size
    )
    
    return train_generator, validation_generator

def plot_model_results(num_epochs, history, key):

    plt.plot(range(num_epochs), history[key])
    plt.plot(range(num_epochs), history['val_' + key])
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.show()

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0,
    horizontal_flip=True, 
    zoom_range=0.2,
    shear_range=0.2
)
NUM_CLASSES = 12
EPOCHS = 200
TARGET_DIM = 84
BATCH_SIZE = 64
MODEL_TRAIN = 'vgg19'
train_generator, validation_generator = get_data_generators(datagen, TARGET_DIM, BATCH_SIZE)
base_model = tf.keras.applications.vgg19.VGG19(
    include_top=False, 
    weights='imagenet', 
    input_shape=(TARGET_DIM, TARGET_DIM, 3)
)
# base_model = tf.keras.applications.inception_v3.InceptionV3(
#     include_top=False, 
#     weights='imagenet', 
#     input_shape=(TARGET_DIM, TARGET_DIM, 3)
# )
# base_model = tf.keras.applications.mobilenet.MobileNet(
#     include_top=False, 
#     weights='imagenet', 
#     input_shape=(TARGET_DIM, TARGET_DIM, 3)
# )
preds = base_model.output
preds = tf.keras.layers.GlobalAveragePooling2D()(preds)
preds = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.BatchNormalization()(preds)
preds = tf.keras.layers.Dense(512, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.BatchNormalization()(preds)
preds = tf.keras.layers.Dense(256, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.BatchNormalization()(preds)
preds = tf.keras.layers.Dense(128, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)(preds)
model = tf.keras.models.Model(base_model.input, preds)
# model.summary()
vgg19 = ModelTrain(model, TARGET_DIM, BATCH_SIZE)
vgg19.freeze_layers(9)
# vgg19.model.summary()

vgg19.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['acc']
)
# Lets define checkpoint for model saving
filepath=f"./checkpoint/{MODEL_TRAIN}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only = False, save_best_only=True, mode='min')
tfboard = TensorBoard(log_dir=f'./logs/{MODEL_TRAIN}_training')
training_checkpoint = TrainingCheckpoint()

csv_logger = tf.keras.callbacks.CSVLogger(filename=f'./logs/{MODEL_TRAIN}_training.csv', append=True)
vgg19.model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, csv_logger, training_checkpoint, tfboard],
    epochs=EPOCHS
)
model_history = vgg19.model.history.history

plot_model_results(EPOCHS, model_history, 'acc')
plot_model_results(EPOCHS, model_history, 'loss')