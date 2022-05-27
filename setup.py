timestick = '04-05-2022'
'''
Init params
'''
SAVE_PATH = 'dataset/dataset12ClassKeras/'
ROOT_PATH = 'dataset/'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
listType = ["train", "val", "test"]
listLabel = ["Cell","Cell-Multi","Cracking","Diode","Diode-Multi","Hot-Spot","Hot-Spot-Multi","No-Anomaly","Offline-Module","Shadowing","Soiling","Vegetation"]
BATCH_SIZE = 64
IMG_HEIGHT = 40
IMG_WIDTH = 24
num_classes = 12
filter = 64 # depth of your CNN model
lr = 0.0003
checkpoint_best = f'checkpoint/{timestick}.hdf5'
logdir = f'logs/{timestick}'
patience_to_stop = 20
epochs = 200
history_best = f'history/{timestick}.npy'