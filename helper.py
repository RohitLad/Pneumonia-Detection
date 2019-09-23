import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def tf_reset_graph(model=None):
    tf.reset_default_graph()
    K.clear_session()
    gc.collect()

def tf_reset_callbacks(checkpoint=None, reduce_lr=None, early_stopping=None, tensorboard=None):
    checkpoint = None
    reduce_lr = None
    early_stopping = None
    tensorboard = None

def create_image_data_generator(target_size, rescale, path, batch_size, class_mode = 'categorical', shuffle=True, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True):

    data_generator = ImageDataGenerator(rescale=rescale,
                                        shear_range=shear_range,
                                        zoom_range=zoom_range,
                                        horizontal_flip=horizontal_flip)

    generator = data_generator.flow_from_directory(path,
                                       class_mode=class_mode,
                                       target_size=target_size,
                                       shuffle=shuffle,
                                       batch_size=batch_size)

    return generator