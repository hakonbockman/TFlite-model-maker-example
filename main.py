import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader
from tflite_model_maker.image_classifier import ModelSpec

import matplotlib.pyplot as plt

import shutil
import sys
import splitfolders
import re



from tflite_model_maker.config import ExportFormat
from tflite_model_maker import image_classifier
#from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec
from tflite_model_maker import config


from pathlib import Path
from enum import Enum
from splitfolders.split import group_by_prefix #split-folders
from tensorflow.keras.preprocessing import image_dataset_from_directory



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

image_path_dict = {
'image_path_IR_combined'                : '../saue bilder/Combined/IR_originale',
'img_path_IR_duplicate_STRICT'          : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_STRICT',
'img_path_IR_duplicate_BASIC'           : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_BASIC',
'img_path_IR_duplicate_LOOSE'           : '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_LOOSE',
'img_path_IR_BLURRY_duplicate_STRICT'   : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_STRICT',
'img_path_IR_BLURRY_duplicate_BASIC'    : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_BASIC',
'img_path_IR_BLURRY_duplicate_LOOSE'    : '../saue bilder/Combined/IR_removed_duplicate_with_blurry/IR_removed_duplicate_LOOSE',
'img_path_VISUAL'                       : '../saue bilder/Combined/Visual',
'img_path_VISUAL_duplicate_STRICT'      : '../saue bilder/Combined/Visual_removed_duplicate_STRICT',
'img_path_VISUAL_duplicate_BASIC'       : '../saue bilder/Combined/Visual_removed_duplicate_BASIC',
'img_path_VISUAL_duplicate_LOOSE'       : '../saue bilder/Combined/Visual_removed_duplicate_LOOSE'
}

# Helper function that returns "red or black" depending on if its two input parameters mathces or not
def get_label_color(val1, val2):
    if val1 == val2:
        return 'black'
    else:
        return 'red'
    
    return None


def plot_classified_images():

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

def pick_and_copy_classified_pictures():
    predicts = model.predict_top_k(test_data)

    for i, (image, label) in enumerate(test_data.dataset):
        print(predicts[i])

    return None

def plot_before_classifing(data):
    plt.figure(figsize=( float(10), float(10) )) # needs to be 2 floats or else it crashes
    for i, (image, label) in enumerate( data.dataset.take(25)):
        plt.subplot(5, 5, i + 1)
        plt.xticks( [] )
        plt.yticks( [] )
        plt.grid(False)
        plt.imshow( image.numpy(), cmap=plt.cm.gray)
        plt.xlabel( data.index_to_label[ label.numpy() ])
    plt.show()

# ef ratio(input, output="output", seed=1337, ratio=(0.8, 0.1, 0.1), group_prefix=None):
def split_train_val_test(path):
    # Creating training set, validation set and test set.  80/10/10 split

    splitfolders.ratio(
        input=str(path), 
        output=str(path+"/split"), 
        seed=1337, 
        ratio=(.8, .1, .1),
        group_prefix=None
        )
    return None

# splitting 80% train, 20% validation
def split_train_val(path):

    splitfolders.ratio(
        input=str(path), 
        output=str(path+"/split"), 
        seed=1337, 
        ratio=(.8, .2),
        group_prefix=None
        )
    return None



### DATA SPLITS 
#train_data, rest_data = data.split(0.8)                 # Training  = 80%
#validation_data, test_data = rest_data.split(0.5)       # Test = 10%  # Validating = 10%



# TODO: need to re-organize pictures into test, validation and training set

def load_all_pictures():
    for key in image_path_dict:
        path = os.path.join(image_path_dict[key])
        try:
            split_path = os.path.join(image_path_dict[key] + "/split")
            #os.mkdir(split_path, "test_dir")
            os.rmdir(split_path, "test_dir")
        except:
            print("....................................NO SPLIT FOLDER........................")
            print("initiate splitting dataset to train, value and test..")
            print("splitting folder: ", str(path))
            print("...")
            split_train_val_test(path)
    return None


def creating_keras_dataset(train_data, validation_data, test_data, batch_size, image_size ):
    if train_data is None:
        print("wrong paths passed over to creating_keras_dataset")
    training_dataset = image_dataset_from_directory(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size
    )

    validation_dataset = image_dataset_from_directory(
        validation_data,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size
    )
    test_dataset = image_dataset_from_directory(
        test_data,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size
    )
    return training_dataset, validation_dataset, test_dataset

''' Function for plotting the first 9 pictures in the training set from the dataset. '''
def show_first_9_pictures_in_dataset(train_dataset, class_names):
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

# TODO Normalize pictures












if __name__ == '__main__':
    # Load all pictures according to dict
    # load_all_pictures()

    # creeating an tflite-model-maker dataset
    #split a datasaet to 80/20 test/validation

    path = os.path.join('../saue bilder/Combined/Visual_originale')
    split_train_val(path)


    batch_size = 32
    image_size = (256, 256)
    # 
    for i, (k, v) in enumerate(image_path_dict.items()):
        training = os.path.join(v + "/split/train")
        validation = os.path.join(v + "/split/val")
        testing = os.path.join(v + "/split/test")


        #TODO: if IR do grayscale option?
        train_dataset, validation_dataset, testing_dataset = creating_keras_dataset(training, validation, testing, batch_size, image_size)
        

        # IMAGE CLASSIFICATION NETWORKS
        # 'efficientnet_lite0', 'efficientnet_lite1', 
        # 'efficientnet_lite2','efficientnet_lite3', 
        # 'efficientnet_lite4', 'mobilenet_v2', 'resnet_50'

      
        model_spec.IMAGE_CLASSIFICATION_MODELS

        #TODO: Problem is that I am making a KERAS dataset and then trying to use Tflite-model-maker method.
        data = DataLoader.from_folder('../saue bilder/Combined/Visual_originale')
        train_dataset, testing_data = data.split(0.9)
       
       
        #TODO: labeled_index = not_sau, sau
        model = image_classifier.create( 
            train_dataset#,           
            #model_spec=model_spec.get('mobilenet_v2'),                              # Model/Architecture/Algorithm used
            #validation_data=validation_dataset,                    
            #batch_size=128,
            #epochs=5,
            #learning_rate=0.0002,
            #dropout_rate= 0.5,
            #shuffle=True
            #train_whole_model=False
            )

        jajaja = model.index_to_label

        model.summary()

        loss, accuracy = model.evaluate(testing_dataset)

        #plot_classified_images()
        pick_and_copy_classified_pictures()

        export = os.path.join(i + "/export")
        #i =

        model.export(
            # Exporting
            export_dir=export, 
            label_filename='sheep_labels',
            export_format=ExportFormat.SAVED_MODEL,
            
            overwrite=True, 
            # Saving
            saved_model_filename='tflite_sheep_classiification_model_saved',
            
        )







# picking out the pitctures that are tested and beilieved to be sheeps
#predicts = model.predict_top_k(test_data)

#for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take()):





#model.evaluate_tflite('model.tflite', batch_size)



####### Graveyard #############

'''
# PATH TO PICTURES


# IR 
# NORMAL
image_path_IR_combined = '../saue bilder/Combined/IR_orignale'

# REMOVED DUPLICATES
img_path_IR_duplicate_STRICT = '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_STRICT'
img_path_IR_duplicate_BASIC = '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_BASIC'
img_path_IR_duplicate_LOOSE = '../saue bilder/Combined/IR_removed_duplicate_withOUT_blurry/removed_duplicate_LOOSE'

# REMOVED DUPLICATES CONTAINS BLURRY PICTURES
img_path_IR_BLURRY_duplicate_STRICT = '../saue bilder/Combined/IR_removed_duplicate_with_blurry/removed_duplicate_STRICT'
img_path_IR_BLURRY_duplicate_BASIC = '../saue bilder/Combined/IR_removed_duplicate_with_blurry/removed_duplicate_BASIC'
img_path_IR_BLURRY_duplicate_LOOSE = '../saue bilder/Combined/IR_removed_duplicate_with_blurry/removed_duplicate_LOOSE'


 VISUAL 
# NORMAL
img_path_visual = '../saue bilder/Combined/Visual'
#C:/Users/hakon/Git/sa
# REMOVED DUPLICATES
img_path_VISUAL_duplicate_STRICT = '../saue bilder/Combined/Visual_removed_duplicate_STRICT'
img_path_VISUAL_duplicate_BASIC = '../saue bilder/Combined/Visual_removed_duplicate_BASIC'
img_path_VISUAL_duplicate_LOOSE = '../saue bilder/Combined/Visual_removed_duplicate_LOOSE'




# LOADING THE IMAGES

#data = ImageClassifierDataLoader.from_folder(img_path_visual)



###  CREATING TENSERFLOW DATASETS
path = os.getcwd()
print("the current directory is:  %s %", path)

os.chdir(IMAGE_PATH_VISUAL)
path = os.getcwd()
print("the NEW directory is: %s %", path)
'''
