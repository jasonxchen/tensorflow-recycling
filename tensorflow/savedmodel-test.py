import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 16
img_height = 192
img_width = 192

model = tf.keras.models.load_model("models/4")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/recycling-symbols-test",
    labels='inferred',
    label_mode='int',
    shuffle=False,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = ["Recyc1", "Recyc2", "Recyc3", "Recyc4", "Recyc5", "Recyc6", "Recyc7"]
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(7):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)

for images, labels in test_ds.take(1):
    for i in range(7):
        print("-------------------------------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:", class_names[np.argmax(predictions[i])])
    break
