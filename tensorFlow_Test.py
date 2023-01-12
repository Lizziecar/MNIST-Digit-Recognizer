import tensorflow.keras as keras
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import cv2

### Reload data
mnist = tf.keras.datasets.mnist # 28 x 28 handwritten number digits (really basic one)
(x_train, y_train) , (x_test, y_test) =mnist.load_data()

### Load model
new_model = tf.keras.models.load_model('num_reader_mnist.model')

### Make a prediction
#predictions_probability = new_model.predict(x_test)
#print(predictions_probability)

# Pick highest probability
#predictions = []
#for prediction in predictions_probability:
#	predictions.append(np.argmax(prediction))

## Test options:
#index = 50
# Data
#plt.imshow(x_test[index], cmap=plt.cm.binary)
#plt.show()

# corresponding prediction:
#print(predictions[index])

### Out-out of sample testing:
image_data = cv2.imread('4.png', cv2.IMREAD_GRAYSCALE)
# since colour, we change to greyscale

#plt.imshow(image_data, cmap='gray')
#plt.show()

# fix resolution
image_data = cv2.resize(image_data, (28, 28))

plt.imshow(image_data, cmap='gray')
plt.show()

image_data = tf.expand_dims(image_data, 0) # make a fake batch

out_of_sample_probabilty = new_model.predict(image_data)
out_of_sample_prediction = np.argmax(out_of_sample_probabilty)

print(out_of_sample_prediction)

