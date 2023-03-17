import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_data(directory, batch_size, image_height, image_width):
    train_dir = os.path.join(directory, 'train')
    test_dir = os.path.join(directory, 'test') 

    train_ds = image_dataset_from_directory(train_dir,
                                        shuffle=True,
                                        validation_split=0.2,
                                        subset="training",
                                        seed=42,
                                        batch_size=batch_size,
                                        image_size=(image_height, image_width))

    validation_ds = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             validation_split=0.2,
                                             subset="validation",
                                             seed=42,
                                             batch_size=batch_size,
                                             image_size=(image_height, image_width))

    test_ds = image_dataset_from_directory(test_dir,
                                       batch_size=batch_size,
                                       image_size=(image_height, image_width)
                                      )

    return train_ds, validation_ds, test_ds