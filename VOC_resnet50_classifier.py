#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:15:25 2022

@author: shahina
"""
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np 
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_dir = input("Enter path of test  dataset:  ")#/home/shahina/Documents/ML/VOC/dataset3/test
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (224,224),
                                            batch_size = 1,
                                            shuffle=False)
test_samples = len(test_set)


# Loading model and predict.
from tensorflow.keras.models import load_model
model_path = input(" Enter path of Resnet50 model : ")#/home/shahina/Documents/ML/VOC/dataset3/voc/VOC_resnet50.h5
model=load_model(model_path)
pred=model.predict(test_set,steps=test_samples,verbose=1)
y_pred=np.argmax(pred,axis=1)

labels = ['Class1', 'Class2', 'Class3', 'Class4','class5']

predictions = [labels[k] for k in y_pred]
print(predictions)

filenames=test_set.filenames
d ={"Filename":filenames,
      "Predictions":predictions}
df = pd.DataFrame(d) 
df.to_csv("result.csv",index=False)
print("Prediction result is stored in result.csv file in present working directory " )
