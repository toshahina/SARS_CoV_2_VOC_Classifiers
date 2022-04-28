#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:35:34 2022

@author: shahina
"""

import warnings
warnings.filterwarnings("ignore")

import os

import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = input("Enter path of train dataset  : ")
val_dir=input("Enter path of validation dataset  : ")

# function to get count of images
def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count += len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count

train_samples =get_files(train_dir)
num_classes=len(glob.glob(train_dir+"/*"))

print(num_classes,"Classes")
print(train_samples,"Train images")

# Preprocessing data.
train_datagen=ImageDataGenerator()

val_datagen=ImageDataGenerator()

# set height and width and color of input image.
img_width,img_height =224,224
input_shape=(img_width,img_height,3)
batch_size =64
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(img_width,img_height),
                                                   batch_size=batch_size)
model = Sequential()
model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))          
model.add(Dense(num_classes,activation='softmax'))
model.summary()

# validation data.
validation_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_height, img_width),batch_size=batch_size)

# Model building to get trained with parameters.
opt=tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
train=model.fit_generator(train_generator,
                          epochs=25,
                          steps_per_epoch=train_generator.samples // batch_size,
                          validation_data=validation_generator,
                          verbose=1)

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()
plt.savefig('accuracy.png')

#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
plt.savefig('loss.png')

# Save entire model with optimizer, architecture, weights and training configuration.
from tensorflow.keras.models import load_model
model.save('SARS_CoV_2_VOC_Net.h5', include_optimizer=True)

# Save model weights.
from tensorflow.keras.models import load_model
model.save_weights('SARS_CoV_2_VOC_Net_weights.h5')

# Loading model and predict.
from tensorflow.keras.models import load_model
model=load_model('SARS_CoV_2_VOC_Net.h5')

#Testing the model

import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix,classification_report

#%matplotlib inline

test_dir= input(" Enter path of test dataset  : " )
img_width,img_height =224,224
input_shape=(img_width,img_height,3)
batch_size =64

test_datagen=ImageDataGenerator()
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=False,
                                                   target_size=(img_width,img_height),
                                                   batch_size=batch_size)
classes = test_generator.class_indices 

score,accuracy =model.evaluate_generator(test_generator,verbose=1)
print("Test loss is {}".format(score))
print("Test accuracy is {}".format(accuracy))

import numpy 

predictions = model.predict(test_generator)

# Get most likely class
predicted_classes = numpy.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys()) 

import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
report = metrics.classification_report(true_classes, predicted_classes, target_names=classes)
print(report) 

print('Confusion Matrix')
print(confusion_matrix(true_classes, predicted_classes))

def plot_confusion_matrix(true_classes, predicted_classes, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    #Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')


    #Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix.png')
    return ax
    
np.set_printoptions(precision=2)

classes = test_generator.class_indices 

# Plotting non-normalized confusion matrix
plot_confusion_matrix(true_classes, predicted_classes, classes, title='Confusion matrix')

from itertools import cycle

from sklearn.metrics import roc_curve, auc

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp
from sklearn.metrics import roc_auc_score

# Compute ROC curve and ROC area for each class
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_classes = 5

for i in range(n_classes):    
    fpr[i], tpr[i], thresh[i] = roc_curve(true_classes, predictions[:,i], pos_label=i)

# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[0], tpr[0], linestyle='--',color='black', label='Class 3 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='pink', label='Class 4 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.ylim([0.96, 1.001])
plt.legend(loc='best')
plt.savefig('Multiclass_ROC',dpi=300);  


roc_auc = dict()
for i in range(n_classes):
    roc_auc[i] = auc(fpr[i], tpr[i])


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw=2
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'blue', 'olive', 'brown', 'black', 'gray', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('All_ROC',dpi=300)
plt.show()
