import numpy as np
import tensorflow as tf

batch_size = 32
img_height = 150
img_width = 150

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols/",
    labels='inferred',
    label_mode='int',
    validation_split=0.1, 
    subset="training",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    # shuffle=True,    # default is True
    seed=123
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols/",
    labels='inferred',
    label_mode='int',
    validation_split=0.1, 
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    # shuffle=True,
    seed=123
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols-test/",
    labels='inferred',
    label_mode='int',
    # color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    # shuffle=False,    # was True
    seed=123
)

# options for optimizing performance (caching)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# defining the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1.0/255),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.Dense(10)
#     tf.keras.layers.Dense(7)
# ])

# newer (and better?) layers
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0/255),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    # validation_data = 
    epochs=15,
    verbose=2
)

# test_loss, test_acc = model.evaluate(val_ds, verbose=2)
# print('\nTest accuracy:', test_acc)

# Prediction
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)
# print(predictions[0])
# print(np.argmax(predictions[0]))
for pred in predictions:
    print("-------------")
    print(pred)
    print(np.argmax(pred))
