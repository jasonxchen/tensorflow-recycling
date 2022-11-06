import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# file paths
data_dir = "data/recycling-symbols"
test_dir = "data/recycling-symbols-test"

# ds settings
batch_size = 64
img_height = 192
img_width = 192

# loading images into datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.1, 
    subset="training",
    seed=39,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.1, 
    subset="validation",
    seed=39,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    shuffle=False,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# pull out the class labels
class_names = train_ds.class_names
print(class_names)

# options for optimizing performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# defining the model layers
num_classes = len(class_names)

# data augmentaion layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.3),
  tf.keras.layers.RandomContrast(0.2),
  tf.keras.layers.RandomZoom(0.1),
])

# function to process input for mobilenet
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# base mobilenet model pretrained with imagenet ds
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# # preprocess for resnet
# preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

# # base resnet model
# base_model = tf.keras.applications.resnet_v2.ResNet50V2(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(img_height, img_width, 3),
#     pooling='avg',
# )

# Feature extraction?
image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# freeze the weights
base_model.trainable = False

# Adding a classification head?
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Defining some trainable layers -- doesn't seem to work
trainable_layers = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
])

# final layer in model
prediction_layer = tf.keras.layers.Dense(num_classes)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# define input tensor and start setting up our model by chaining together all the defined layers
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs) 
x = preprocess_input(x)

x = base_model(x, training=False) # make sure training is set to false
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.5)(x)
# x = trainable_layers(x)
# x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# lower learning rate for transfer learning is better
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()

# training the model
epochs = 300

# Early stopping to prevent overtraining
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='auto',
    verbose=1,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
)

# Fine tuning by training the entire model
base_model.trainable=True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/100),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()

model.fit(
    train_ds,
    epochs=int(epochs/2),
    validation_data=val_ds,
    callbacks=callbacks
)

# File path for saving model
saved_model = "models/6" # increment number for new versions

# Saving the trained model
tf.keras.models.save_model(
    model=model,
    filepath=saved_model,
    overwrite=True
)

# Test model against validation ds
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print('\nTest accuracy:', test_acc)

# Predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)

for images, labels in test_ds.take(1):
    for i in range(7):
        print("-------------------------------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:", class_names[np.argmax(predictions[i])])
    break

# Plotting training graph
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

# Making sure everything is cleaned up (might not be neccessary)
tf.keras.backend.clear_session()
