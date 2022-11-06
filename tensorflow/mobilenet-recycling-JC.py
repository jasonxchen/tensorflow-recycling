import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Custom dataset
data_dir = "data/recycling-symbols-filtered"
test_dir = "data/recycling-symbols-test"
batch_size = 8
img_height = 224
img_width = 224
class_names = ["Recyc1", "Recyc2", "Recyc3", "Recyc4", "Recyc5", "Recyc6", "Recyc7"]

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# plt.figure(figsize=(10, 10))
# for images, labels in val_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()


# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

mobile = tf.keras.applications.mobilenet.MobileNet()
# x = mobile.layers[-5].output    # include up to 5th to last layer
x = mobile.layers[-3].output    # include up to 3rd to last layer
x = tf.keras.layers.Reshape(target_shape=(1000,))(x)    # was 1024
output = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs=mobile.inputs, outputs=output)

# only train with last 24 layers (adjust as needed)
for layer in model.layers[:-24]:    # 24 includes extra pooling + dropout from mobilenet
    layer.trainable = False
    
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    # "if softmax layer is not being added to the last layer then we need to have from_logits=True"
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    # loss="categorical_crossentropy",
    metrics=["accuracy"]
)

epochs=10
history = model.fit(
    x=train_ds,
    steps_per_epoch=len(train_ds),
    validation_data=val_ds,
    validation_steps=len(val_ds),
    epochs=epochs,
    verbose=2
)


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)
for images, labels in test_ds.take(1):
    for i in range(7):
        print("-------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:", class_names[np.argmax(predictions[i])])
