#!/usr/bin/python3
import matplotlib.pyplot as plt
import sys
import os
midx = 0
midy = 0
x1 = 0
x2 = 1
y1 = 0
y2 = 1
midx = (x1 + x2)/2
midy = (y1 +y2)/2

direct_name = input('Enter the path of Fasta files: ')
foldername = input('Enter the path of a folder to store CGR images: ')

for filename in os.listdir(direct_name):

 fname = filename.split('.')[0]
 print("Drawing CGR of " + fname)

 with open(direct_name + "/" + filename, 'r') as reader:
  xi = 0
  yj = 0
  xi_prev = midx
  yj_prev = midy
 
  fig = plt.figure(filename)
  plt.plot([x1,x2],[y1,y1],color = "black")
  plt.plot([x2,x2],[y1,y2],color = "black")
  plt.plot([x2,x1],[y2,y2],color = "black")
  plt.plot([x1,x1],[y2,y1],color = "black")
  
  for line in reader.readlines():
   line.rstrip()
   if line.startswith(">"):
    print(fname)
   else :
    for s in line :
   
     if s in ["A","a"]:
      xi = (0.5*(xi_prev + x1))
      yj = (0.5*(yj_prev + y1))
      plt.scatter(xi,yj,s=2,color = "black")
     
     elif s in ["T","t"]:
      xi = (0.5*(xi_prev + x2))
      yj = (0.5*(yj_prev + y1))
      plt.scatter(xi,yj,s=2,color='black')
     
     elif s in ["G","g"] :
      xi = (0.5*(xi_prev + x2))
      yj = (0.5*(yj_prev + y2))
      plt.scatter(xi,yj,s=2, color='black')
     
     elif s in ["C","c"]:
      xi = (0.5*(xi_prev + x1))
      yj = (0.5*(yj_prev + y2))
      plt.scatter(xi,yj,s=2,color = "black")
     
     xi_prev = xi
     yj_prev = yj
   line = ""
   t = ""
  plt.gca().set_axis_off()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0, 0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(foldername + fname + ".png")  
  print(fname + " " + "finished")
  plt.close(fig)
