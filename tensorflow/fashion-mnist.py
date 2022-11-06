# https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf
import numpy as np

print(tf.__version__)

# Import fashion mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# for development use to convert categorical integers to semantic labels
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)

# Preprocess data: 0=black 255=white -> 0=black 1=white
train_images = train_images / 255.0
test_images = test_images / 255.0

# Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make PREDICITONS
# "Attach a softmax layer to convert the model's linear outputs—logits—to probabilities"
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# since test_images has 10k images, predictions will be an array containing 10k arrays, each having 10 values representing the probabilities that an item is one of the 10 categories
predictions = probability_model.predict(test_images)
# return the highest confidence value
# np.argmax(predictions[0])
print(np.argmax(predictions[0]))
# ^ should return "9", meaning the first image in test_images is predicted to be in the category 9 => Ankle boot (which is correct)
