import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import ocnfigs
from tflite_model_maker import ExportFormat
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec

import matplotlib as plt


image_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

data = ImageClassifierDataLoader.from_folder(image_path)

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)


plt.figure(figsize=(10,10))

for i, (image, label) in enumerate( data.dataset.take(25)):
    plt.subplot(5, 5, i + 1)
    plt.xticks( [] )
    plt.yticks( [] )
    plt.grid(False)
    plt.imshow( image.numpy(), cmap=plt.cm.gray)
    plt.xlabel( data.index_to_label[ label.numpy() ])
plt.show()



model = image_classifier.create( train_data, validation_data=validation_data)

model.summary()

loss, accuracy = model.evaluate(test_data)





model.export(export_dir='/export/')