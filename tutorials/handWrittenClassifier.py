import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# Import the dataset
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

#Display some digits
def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)

# Visualize the digits
display(0)
# visualize how many features we have
print(len(data[0]))

# Create a Linear Classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=10000)

# Evaluate the accuracy
classifier.evaluate(test_data, test_labels)
print(classifier.evaluate(test_data, test_labels)['accuracy'])

# Visualize learned weights
f, axes = plt.subplots(2, 5, figsize=(10,4))
weights = classifier.get_variable_value("linear//weight/d/linear//weight/part_0/Ftrl_1")
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(()) # ticks be gone
    a.set_yticks(())
plt.show()

