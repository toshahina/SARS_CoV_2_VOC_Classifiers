#!/usr/bin/python3
import sys
import os
import shutil

direct_name = input('Enter the fasta file directory name: ')
foldername = input("Enter a folder name to store the N containing sequence_files:")
num_n = 0
for filename in os.listdir(direct_name):

 fname = filename.split('.')[0]
 print("Drawing CGR of " + fname)

 with open(direct_name + "/" + filename, 'r') as reader:
  
  for line in reader.readlines():
   line.rstrip()
   if line.startswith(">"):
    print(fname)
   else :
    for s in line :
     if s in ["N","n"]:
      num_n = num_n + 1
      
  print( "Number of N's containing files : ",num_n)
  if(num_n!=0):
   shutil.move(direct_name + "/" + filename, foldername + "/" + filename)
  num_n = 0
