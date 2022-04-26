#!/usr/bin/python3
#import matplotlib.pyplot as plt
import sys
import os
filename = input("Enter fasta filename containing multiple sequences : ")
path = input("Enter path of diectory name to store individual fasta files: ")
newname = open("txt1.txt","w")
with open(filename, 'r') as reader:
  for line in reader.readlines():
   line.rstrip()
   if line.startswith(">"):
    newname.close()
    fname1 = line.split("|")
    fname2 = fname1[0].split(">")
    fname3 = fname2[1]
    fname4 = fname3.split("/")
    s = "_"
    fname = s.join(fname4)
    print(fname)
    newname = open(path + "/" + fname + ".fasta","w")
    newname.write(line)
   else :
    newname.write(line)
  
