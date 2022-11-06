import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import
batch_size = 2
img_height = 200
img_width = 200
train_images = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols-filtered/",
    labels='inferred',
    label_mode='int',
    validation_split=0.3,
    subset="training",
    seed=1234,
    # shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_images = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols-filtered/",
    labels='inferred',
    label_mode='int',
    validation_split=0.3, 
    subset="validation",
    seed=1234,
    # shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# Grid plot of first x images in val_images
class_names = ["Recyc1", "Recyc2", "Recyc3", "Recyc4", "Recyc5", "Recyc6", "Recyc7"]
# plt.figure(figsize=(10, 10))
# for images, labels in val_images.take(1):
#   for i in range(2):    # range(x)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()


# speed up with caching (no difference tbh)
AUTOTUNE = tf.data.AUTOTUNE
train_images = train_images.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_images = val_images.cache().prefetch(buffer_size=AUTOTUNE)

# Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    # tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.1),    # 0.2???
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
epochs=20
history = model.fit(train_images, validation_data=val_images, epochs=epochs)

test_loss, test_acc = model.evaluate(val_images, verbose=2)
print('\nTest accuracy:', test_acc)


# Visualize graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Make PREDICITONS
test_images = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols-test/",
    labels='inferred',
    label_mode='int',
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=7
)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


# return the highest confidence value
# np.argmax(predictions[0])
# print(np.argmax(predictions[0]))

# for i in range(9):
#     print("------------")
#     print(np.argmax(predictions[i]))

for images, labels in test_images.take(1):
    for i in range(7):
        print("-------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:", class_names[np.argmax(predictions[i])])
