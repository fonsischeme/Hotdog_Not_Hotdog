import os
import time

import tensorflow as tf

def set_up_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def get_run_logdir(model):
    root_logdir = os.path.join(os.curdir, "logs","{}".format(model))
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)