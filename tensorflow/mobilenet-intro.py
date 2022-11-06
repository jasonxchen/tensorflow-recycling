import numpy as np
import tensorflow as tf
from tensorflow import keras
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = 'data/MobileNet-samples/'    # sample
    # img_path = 'data/recycling-symbols-test/1/'    # try
    img = tf.keras.preprocessing.image.load_img(img_path + file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

from IPython.display import Image
Image(filename='data/MobileNet-samples/1.jpeg', width=300,height=200)    # sample
# Image(filename='data/recycling-symbols-test/1/IMG_6745.jpg', width=200,height=200)    # try

preprocessed_image = prepare_image('1.jpeg')    # sample
# preprocessed_image = prepare_image('IMG_6745.jpg')    # try
predictions = mobile.predict(preprocessed_image)

results = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
print(results)
