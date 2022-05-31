import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotLearningCurve(history):

  plt.plot(history.history['accuracy'], color='red')
  plt.plot(history.history['val_accuracy'], color='blue')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

  plt.plot(history.history['loss'], color='red')
  plt.plot(history.history['val_loss'], color='blue')
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()



def train_val_generators(DIR):

  train_datagen = ImageDataGenerator(rescale = 1/255.0, validation_split=0.20)

  train_generator = train_datagen.flow_from_directory(directory=DIR,
                                                      batch_size=64, 
                                                      class_mode='binary',
                                                      target_size=(120, 120),
                                                      shuffle=True,
                                                      subset='training')


  validation_generator = train_datagen.flow_from_directory(directory=DIR,
                                                                batch_size=64, 
                                                                class_mode='binary',
                                                                target_size=(120, 120),
                                                                subset='validation')


  test_datagen = ImageDataGenerator(rescale = 1/255.0)


  test_generator = test_datagen.flow_from_directory(directory='Test/',
                                                                batch_size=1, 
                                                                class_mode='binary',
                                                                target_size=(120, 120),
                                                                shuffle=False)


  return train_generator, validation_generator, test_generator


  def train_val_generators_data_aug(DIR):
  
    
    train_datagen = ImageDataGenerator(
                                          rescale = 1./255., 
                                          rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest',
                                          validation_split=0.2
                                      )

    train_generator = train_datagen.flow_from_directory(directory=DIR,
                                                        batch_size=64, 
                                                        class_mode='binary',
                                                        target_size=(120, 120),
                                                        shuffle=True,
                                                        subset='training')


    validation_generator = train_datagen.flow_from_directory(directory=DIR,
                                                                  batch_size=64, 
                                                                  class_mode='binary',
                                                                  target_size=(120, 120),
                                                                  shuffle = True,
                                                                  subset='validation')


    test_datagen = ImageDataGenerator(rescale = 1/255.0)


    test_generator = test_datagen.flow_from_directory(directory='Test/',
                                                                  batch_size=1, 
                                                                  class_mode='binary',
                                                                  target_size=(120, 120),
                                                                  shuffle=False)


  return train_generator, validation_generator, test_generator


def plot_roc(fpr,tpr, label_=None):

    plt.plot(fpr,tpr,label=label_)
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
