# https://www.youtube.com/watch?v=q7ZuZ8ZOErE
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial18-customdata-images/1_in_subfolders.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

# set parameters of dataset
img_height = 28
img_width = 28
batch_size = 2
model = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(16, 3, padding="same"),
        keras.layers.Conv2D(32, 3, padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
])

# Method 1
# make training set from custom-mnist (shuffled data)
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "data/custom-mnist/",
    labels="inferred",
    label_mode="int",  # alternatives: categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),    # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.3,
    subset="training",
)
# make validation set from custom-mnist with same shuffling seed and split
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "data/custom-mnist/",
    labels="inferred",
    label_mode="int",
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),    # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.3,
    subset="validation",
)
# make test set from custom-mnist-test
ds_test = tf.keras.utils.image_dataset_from_directory(
    "data/custom-mnist-test",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False,
)
# Compile and Train
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(ds_train, epochs = 10, verbose = 2)

test_loss, test_acc = model.evaluate(ds_validation, verbose=2)
print('\nTest accuracy:', test_acc)

# Predict
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(ds_test)
print(predictions[0])
print(np.argmax(predictions[0]))
# ^ should return "2", meaning the first/only image in custom-mnist-test is predicted to be the number 2
# Note: with small dataset of 49 images to train on, predictions are not consistently accurate

# Method 2
# [...]
