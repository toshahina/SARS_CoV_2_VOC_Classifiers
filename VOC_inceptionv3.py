#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:04:08 2022

@author: shahina
"""

import math
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix

import itertools 
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

inception_v3 = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception_v3.layers:
    layer.trainable = False
train_dir = input("Enter path of training dataset : ")
validation_dir = input (" Enter path of validation dataset : ")
test_dir = input (" Enter path of test dataset : ")

folders = glob(train_dir + "/*")
print("output:",folders)
x = Flatten()(inception_v3.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception_v3.input, outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

batch_size = 64

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

validation_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(224, 224),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')
validation_set = validation_datagen.flow_from_directory(validation_dir,
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (224, 224),
                                            batch_size = 1,
                                            class_mode = 'categorical', 
                                            shuffle=False)
print("Training set",len(training_set))
print("Validation set",len(validation_set))
nb_train_samples = len(training_set)
nb_validation_samples = len(validation_set)
nb_test_samples= len(test_set)

compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))

steps_per_epoch = compute_steps_per_epoch(nb_train_samples)
val_steps = compute_steps_per_epoch(nb_validation_samples)
print("valsteps is : ", val_steps)
epochs= 25

r = model.fit(
  training_set,
  validation_data=validation_set,
  epochs=epochs,
  steps_per_epoch=steps_per_epoch,
  validation_steps=val_steps
)

# Save entire model with optimizer, architecture, weights and training configuration.
from tensorflow.keras.models import load_model
model.save('voc_inception.h5', include_optimizer=True)
model.save_weights('voc_inception_weights.h5')

test_score = model.evaluate(test_set, batch_size)
print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100)) 
print("[INFO] Loss: ",test_score[0])

# plot the loss curves
plt.figure(figsize=[8,6])

plt.plot(r.history['loss'],'r',linewidth=3.0)

plt.plot(r.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)
plt.savefig('loss.png')

# Plot the Accuracy Curves

plt.figure(figsize=[8,6])

plt.plot(r.history['accuracy'],'r',linewidth=3.0)

plt.plot(r.history['val_accuracy'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
plt.savefig('Accuracy.png') 


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('Confusion Matrix.png')
    return ax
    
np.set_printoptions(precision=2)

#Print the Target names
target_names = []
for key in training_set.class_indices:

    target_names.append(key)

pred = model.predict(test_set,batch_size = 1)
y_pred = np.argmax(pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y_pred)
print(cm)

plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

print('Classification Report')
print(classification_report(test_set.classes, y_pred, target_names=target_names))

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
    fpr[i], tpr[i], thresh[i] = roc_curve(test_set.classes, pred[:,i], pos_label=i)
    
    
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
plt.savefig('Multiclass_ROC',dpi=300) 


roc_auc = dict()
for i in range(n_classes):
    roc_auc[i] = auc(fpr[i], tpr[i])


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw=2
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

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
plt.savefig('All_ROC',dpi=300); 
plt.show()

