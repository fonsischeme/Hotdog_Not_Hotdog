import argparse
import sys

import tensorflow as tf
from tensorflow import keras

import data
import models
import utils

def main(args):
    # Globl Variables
    model_type = args.model_type.lower()
    image_width = 256
    image_height = 256
    channels = 3

    # Checking if Valid Model
    valid_models = ['vgg16', 'vgg19', 'alexnet', 'resnetv2', 'inceptionv3', 'resnet50']
    if model_type not in valid_models:
        print("Not valid model. Models that can be used are the follow:\n\t* {}".format("\n\t* ".join(valid_models)))
        sys.exit()

    # Checking for GPUs
    utils.set_up_gpu()

    # Get Data
    train_ds, validation_ds, test_ds = data.load_data(args.directory, 
                                                      args.batch_size, 
                                                      image_height,
                                                      image_width)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    ## Initializing Model
    if model_type == 'vgg16':
        model = models.create_VGG16(image_height, image_width, channels, args.verbose)
    elif model_type == 'vgg19':
        model = models.create_VGG19(image_height, image_width, channels, args.verbose)
    elif model_type == 'alexnet':
        model = models.create_AlexNet(image_height, image_width, channels, args.verbose)        
    elif model_type == 'resnetv2':
        model = models.create_ResNetV2(image_height, image_width, channels, args.verbose)
    elif model_type == 'resnet50':
        model = models.create_ResNet50(image_height, image_width, channels, args.verbose)
    elif model_type == 'inceptionv3':
        model = models.create_InceptionV3(image_height, image_width, channels, args.verbose)

    # Compile Model
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  metrics=['accuracy',
                           tf.keras.metrics.AUC(), 
                           tf.keras.metrics.Recall(),
                          ]
                )

    # Set Up Tensorboard
    run_logdir = utils.get_run_logdir(model_type)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    # Set up Early Exit Criteria
    early_exit_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(
              train_ds,
              validation_data=validation_ds,
              epochs=args.epoch,
              callbacks=[early_exit_cb, tensorboard_cb]
              )
    
    model.evaluate(test_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", action="store", 
                        help='Type Of Model That Will Train'
                        )
    parser.add_argument("directory", action="store", 
                        help='Path to where data is stored'
                        )
    parser.add_argument("-b", "--batch_size", action="store", type=int, default=32, 
                        help='Number of examples in each batch'
                        )
    parser.add_argument("-e", "--epoch", action="store", type=int, default=1,
                        help='Number of Epochs Model will train for'
                        )
    parser.add_argument("-l", "--learning_rate", action="store", type=float, default=0.001,
                        help="The model's learning rate"
                        )
    parser.add_argument("-v", "--verbose", action='store_true',
                        help='Execute verbosely'
                        )
    main(parser.parse_args())