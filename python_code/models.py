import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet50, VGG16, VGG19
from tensorflow.keras.models import Model

def create_AlexNet(image_height, image_width, channels, verbose=False):
    # Create Model
    inputs = keras.Input(shape=(image_height, image_width, channels), name='image')

    x = keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), strides=2)(x)
    x = keras.layers.Conv2D(256, (5, 5), strides=1, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), strides=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(384, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(384, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(.5)(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    # Create a New Model
    model = tf.keras.Model(inputs, outputs, name='AlexNet')
    if verbose:
        model.summary()

    return model

def create_InceptionV3(image_height, image_width, channels, verbose=False):
    # Loading Model
    pretrained_model = InceptionV3(weights = 'imagenet', 
                                   include_top = False, 
                                   input_shape=(image_height, 
                                                image_width, 
                                                channels)
                                                  )
    if verbose:
            pretrained_model.summary()
        
    # Freezing the layers
    pretrained_model.trainable = False

    # Modification of pretrained model
    last_layer = pretrained_model.get_layer('mixed10')
    last_output = last_layer.output
    
    x = keras.layers.GlobalMaxPooling2D()(last_output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    # Creating a new model
    model = Model(pretrained_model.input, x)
    if verbose:
        model.summary()

    return model

def create_ResNetV2(image_height, image_width, channels, verbose=False):
    # Loading Model
    pretrained_model = InceptionResNetV2(weights = 'imagenet', 
                                         include_top = False, 
                                         input_shape=(image_height, 
                                                     image_width, 
                                                     channels)
                                         )
    if verbose:
            pretrained_model.summary()
        
    # Freezing the layers
    pretrained_model.trainable = False

    # Modification of pretrained model
    last_layer = pretrained_model.get_layer('conv_7b_ac')
    last_output = last_layer.output
    
    x = keras.layers.GlobalMaxPooling2D()(last_output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    # Creating a new model
    model = Model(pretrained_model.input, x)
    if verbose:
        model.summary()

    return model

def create_ResNet50(image_height, image_width, channels, verbose=False):
    # Loading Model
    pretrained_model = ResNet50(weights = 'imagenet', 
                                include_top = False, 
                                input_shape=(image_height, 
                                            image_width, 
                                            channels)
                                )
    if verbose:
            pretrained_model.summary()
        
    # Freezing the layers
    pretrained_model.trainable = False

    # Modification of pretrained model
    last_layer = pretrained_model.get_layer('conv5_block3_out')
    last_output = last_layer.output
    
    x = keras.layers.GlobalMaxPooling2D()(last_output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    # Creating a new model
    model = Model(pretrained_model.input, x)
    if verbose:
        model.summary()

    return model

def create_VGG16(image_height, image_width, channels, verbose=False):
    # Loading Model
    pretrained_model = VGG16(input_shape=(image_height, 
                                          image_width,
                                          channels),
                             include_top=False,
                             weights="imagenet"
                             )
    if verbose:
        pretrained_model.summary()
    
    # Freezing the layers
    pretrained_model.trainable = False

    # Modification of pretrained model
    last_layer = pretrained_model.get_layer('block5_pool')
    last_output = last_layer.output
    
    x = keras.layers.GlobalMaxPooling2D()(last_output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    # Creating a new model
    model = Model(pretrained_model.input, x)
    if verbose:
        model.summary()

    return model

def create_VGG19(image_height, image_width, channels, verbose=False):
    # Loading Model
    pretrained_model = VGG19(input_shape=(image_height, 
                                          image_width,
                                          channels),
                             include_top=False,
                             weights="imagenet"
                             )
    if verbose:
        pretrained_model.summary()
    
    # Freezing the layers
    pretrained_model.trainable = False

    # Modification of pretrained model
    last_layer = pretrained_model.get_layer('block5_pool')
    last_output = last_layer.output
    
    x = keras.layers.GlobalMaxPooling2D()(last_output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    # Creating a new model
    model = Model(pretrained_model.input, x)
    if verbose:
        model.summary()

    return model