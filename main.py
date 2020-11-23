import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec

import matplotlib.pyplot as plt

def plot_classified_images(self):

    # Helper function that returns "red or black" depending on if its two input parameters mathces or not
    def get_label_color(val1, val2):
        if val1 == val2:
            return 'black'
        else:
            return 'red'

    # Then plot 100 test images and their predicted labels. If a prediciton result is 
    # different from the label provided label in "test" dataset, we will highligh it in red color.
    plt.figure(figsize=(20.0, 20.0)) # Needs to force Float numbers.
    predicts = model.predict_top_k(test_data)
    for i, (image, label) in enumerate( test_data.dataset.take(100)):
        ax = plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)

        predict_label = predicts[i][0][0]
        color = get_label_color(predict_label, test_data.index_to_label[label.numpy()])
        ax.xaxis.label.set_color(color)
        plt.xlabel('Predicted: %s' % predict_label)
    plt.show()

def plot_before_classifing(self, data):
    plt.figure(figsize=( float(10), float(10) )) # needs to be 2 floats or else it crashes
    for i, (image, label) in enumerate( data.dataset.take(25)):
        plt.subplot(5, 5, i + 1)
        plt.xticks( [] )
        plt.yticks( [] )
        plt.grid(False)
        plt.imshow( image.numpy(), cmap=plt.cm.gray)
        plt.xlabel( data.index_to_label[ label.numpy() ])
    plt.show()


# PATH TO PICTURES
#image_path_flowers = tf.keras.utils.get_file( 'flower_photos',
#    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',  untar=True)


#image_path_storlidalen_21_08_2019 = './Storlidalen 21-22 08 2019/Storlidalen 21-22 08 2019/'

image_path_IR_combined = '../Saue_Bilder/Combined/IR' 
image_path_VISUAL_combined = '../Saue_Bilder/Combined/Visual' 



# LOADING THE IMAGES
data = ImageClassifierDataLoader.from_folder(image_path_VISUAL_combined)

# TODO Normalize pictures



### DATA SPLITS 
train_data, rest_data = data.split(0.8)                 # Training  = 80%
validation_data, test_data = rest_data.split(0.5)       # Test = 10%  # Validating = 10%





    

model = image_classifier.create( 
    train_data,                                         # Training Data
    model_spec=model_spec.efficientnet_lite1_spec,      # Model/Architecture/Algorithm used
    validation_data=validation_data,                     # Validation data
    #batch_size=2,
    #'''
    #train_whole_model=True,
    #learning_rate=0.0002,
    #'''
    #epochs=300,
    #shuffle=True,
    #dropout_rate=0.5
   
    )



model.summary()

loss, accuracy = model.evaluate(test_data)



model.export(
    # Exporting
    export_dir='/export', 
    label_filename='/export/sheep_labels',
    export_format=ExportFormat.LABEL,
    
    overwrite=True, 
    # Saving
    saved_model_filename='/export/tflite_sheep_classiification_model_saved',
    
)

model.evaluate_tflite('model.tflite', test_data)

