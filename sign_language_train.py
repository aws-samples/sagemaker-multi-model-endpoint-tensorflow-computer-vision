from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import argparse
import os


# Set Log Level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser() 
    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--output', type=str, default=os.environ.get('SM_CHANNEL_OUTPUT'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_known_args()


if __name__ == '__main__':
    print(f'Using TensorFlow version: {tf.__version__}')
    args, _ = parse_args()
    epochs = args.epochs
    device = '/cpu:0'
    with tf.device(device):
        # Load Data
        train_path = args.train
        validation_path = args.val
        train_batches = ImageDataGenerator().flow_from_directory(train_path, 
                                                                 target_size=(224, 224), 
                                                                 batch_size=10)
        validation_batches = ImageDataGenerator().flow_from_directory(validation_path,
                                                                      target_size=(224,224), 
                                                                      batch_size=30)
        
        # Load Base Model and Freeze Classification Layers
        base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (224,224, 3), pooling='avg')
        for layer in base_model.layers[:-5]:
            layer.trainable = False
            
        # Define a new Model
        last_layer = base_model.get_layer('global_average_pooling2d')
        last_output = last_layer.output
        x = Dense(10, activation='softmax', name='softmax')(last_output)
        new_model = Model(inputs=base_model.input, outputs=x)
        
        # Compile new Model
        new_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        new_model.fit(train_batches, 
                      steps_per_epoch=18, 
                      validation_data=validation_batches, 
                      validation_steps=3, 
                      epochs=epochs, 
                      verbose=1, 
                      callbacks=[])
        
        # Save Model
        new_model.save(f'{args.model_dir}/1')
