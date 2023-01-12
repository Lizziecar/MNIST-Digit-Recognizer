import tensorflow.keras as keras
import tensorflow as tf 
import matplotlib.pyplot as plt

#print(tf.__version__)

### Loading data
mnist = tf.keras.datasets.mnist # 28 x 28 handwritten number digits (really basic one)
(x_train, y_train) , (x_test, y_test) =mnist.load_data()

# print(x_train[0]) # printing first example

### Displaying Data
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

#print(y_train[0]) # printing the answer of [0]

### Normalize Data
# scale from 0 to 1 or -1 to 1

x_train = tf.keras.utils.normalize(x_train, axis=1) # normalize data
x_test = tf.keras.utils.normalize(x_test, axis=1) # uses built in keras function

#print(x_train[0]) # printing first example

### Build Model
model = tf.keras.models.Sequential() # create a sequential mode

# add layers
model.add(tf.keras.layers.Flatten()) # flatten layer to flatten 2d array

# dense layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # dense layer with 128 neurons and relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer with 10 neurons (0-9), softmax activation

# Compile model
model.compile(optimizer='adam',
			  loss = 'sparse_categorical_crossentropy',
			  metrics=['accuracy'])

# adam optimizer, sparse categorical crossentropy loss, shows accuracy

### Training
model.fit(x_train, y_train, epochs=3)


### Validation
val_loss, val_acc = model.evaluate(x_test, y_test) # evaluate is like test the model
print(f' Validation Loss: {val_loss}')
print(f' Validation Accuracy: {val_acc}')

## Save model
model.save('num_reader_mnist.model')

