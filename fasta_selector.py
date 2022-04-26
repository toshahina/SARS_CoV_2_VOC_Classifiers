import numpy as np
import os
import shutil
import random


path = input("Enter the path to fasta sequence files : ")
foldername = input("Enter folder name to store selected number of fasta files : ")
files = os.listdir(path)
k = int(input("Enter number of sequences to move : "))
random_files = random.sample(files, k)
i =0
for x in random_files:
 shutil.move(path + "/" + x, foldername + "/" + x)
 i = i+1
print("Number of files copied : ",i)
