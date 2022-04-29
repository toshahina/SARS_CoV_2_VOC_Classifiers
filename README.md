# SARS_CoV_2_VOC_Classifiers
This directory contains the source code, model and classifier of SARS-CoV2-VOC-Net - a  tool for predicting SARS-CoV-2 variants of concern, which uses deep convolutional neural networks . It also contains source code and classifiers of seven other SARS-CoV-2-VOC classifiers developed using transfer learning models VGG16, VGG19, ResNet50, InceptionV3, Exception, InceptionResNetV2 and MobileNetV2.</br>

The Jupyter Notebook form of all these programs are also available in the directory.</br>

## Prerequisites
The method was implemented in Python with the use of Keras library running at the top of TensorFlow 2.4.1 (using Anaconda Navigator and Spyder 4.2.1)

## Contents present in SARS_CoV_2_VOC_Classifiers
<b>Fasta_separator.py</b>: A python program to split fasta sequences present in a single file. The input is a fasta file and output is a directory containing multiple fasta file. Specify the location of input file and output directory.</br>
<b>Fasta_N_separator.py</b> : A python program to filter fasta sequences contains base “N”. The input is a directory of fasta files containing both N and non N sequences and output is a directory contains non N sequences. Specify the input directory and directory to store N containing sequences. The non N sequences are present in the input directory.</br>
<b>Fasta_selector.py</b> : A python program to select a particular number of fasta files. Input is a directory containing large number of fasta files and output directory contains selected number of fasta files. Provide the location of both input directory and output directory.</br>
<b>CGR_generator.py</b>: A python program to generate CGR images of genomic sequences. Input is a directory containing fasta files and output is CGR images of corresponding images. Give the location of both input and output directory.</br>
<b>SARS_CoV_2_VOC_Net.py, VGG16_voc.py, VGG19_voc_py, Resnet50_voc.py, Inceptionv3_voc.net, Exception_voc.py, InceptionResnetv2_voc.py, Mobilenet_voc.py</b> : These are python programs used to built SARS-CoV-2 variants of concern predictor models using deep learning and transfer learning respectively. Input to the above program is a total of 25000 CGR images of SARS-CoV-2-VOC which are split into train dataset (70% - 17,500 images), validation dataset(10% - 2500 images) and test dataset (20% - 5000 images). The dataset is available in the location given below.</br>
https://drive.google.com/drive/folders/1eMGqCvBgjW9So3pMNsImQr0Y48vWkF4n?usp=sharing </br>

Download the data set and give the location of train directory, validation directory and test directory during program execution. The output is a deep learning model which can predict the SARS-CoV-2 variants of concern. It also output graphs such as Train Accuracy vs Validation Accuracy graphs, Train loss vs validation loss, confusion matrix, classification report and ROC curve of the model in the present working directory. 

<b>SARS_CoV_2_VOC-Net.h5 </b>: Deep learning model developed for classifying SARS-CoV-2-VOC.</br>
<b>VGG16_voc.h5, VGG19_voc.h5, resnet50_voc.h5, inceptionv3_voc.h5, exception_voc.h5, inceptionresnetv2_voc.h5, mobilenet_voc.h5</b> : These are  deep learning models developed for classifying SARS-CoV-2-VOC using VGG16_voc.py, VGG19_voc_py, Resnet50_voc.py, Inceptionv3_voc.py, Exception_voc.py, InceptionResnetv2_voc.py and Mobilenet_voc.py respectively.</br>
<b>SARS_CoV_2_VOC_Net_classifier.py </b>:SARS-CoV-2 variants of concern classifier/predictor</br>
<b>VGG16_voc_classifier.py, VGG19_voc_classifier.py, resnet50_voc_classifier.py, inceptionv3_voc_classifier.py, exception_voc_classifier.py, inceptionresnetv2_voc_classifier.py, mobilenet_voc_classifier.py</b>: SARS-CoV-2 variants of concern classifier using transfer learning models</br>. 
The transfer learning models (.h5) are available in the location given below</br>
https://drive.google.com/drive/folders/18Twu_uKkm26JyuBPkSO10rOS7RY5tpRD?usp=sharing</br>

## How to predict SARS-CoV-2-VOC  using different models.
1. Download fasta sequence of SARS-CoV-2 (GISAID, NCBI Virus etc).
2. Preprocess fasta sequences (Remove sequences with N using Fasta_N_separator.py)
3. Generate CGR images (Using CGR_generator.py)
4. Choose a classifier (SARS_CoV_2_VOC_Net_classifier.py, VGG16_voc_classifier.py, VGG19_voc_classifier.py, resnet50_voc_classifier.py, inceptionv3_voc_classifier.py, exception_voc_classifier.py, inceptionresnetv2_voc_classifier.py, mobilenet_voc_classifier.py and specify the location of the selected model)
5. Predict using the selected classifier
