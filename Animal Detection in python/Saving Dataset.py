from PIL import Image
import numpy as np
import os
import cv2

data=[]
labels=[]
count = 0
path = os.getcwd()
cats=os.listdir(path+'/Cats/') # Make a Folder called 'Cats' in the same 
                               # directory as the main file with images to be trained upon 

'''for i in range(7000, 7010 , 1):# Use this module to sort our the erronous images
    try:
        imag = cv2.imread(path+'/Dogs/'+str(i)+".jpg")
        #time.sleep(0.1)
        if(i%1 == 0):
            print(path+'/Dogs/'+str(i)+".jpg")
    except:
        print("File not Found",path+'/Dogs/'+str(i)+".jpg")
        pass'''

for cat in cats:
    imag = cv2.imread(path+"/Cats"+"/"+cat)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)
    #count+=1
    #print(count)
    
        
dogs=os.listdir(path+'/Dogs') # Make a Folder called 'Dogs' in the same directory 
                              #as the main file with images to be trained upon
for dog in dogs:
    imag= cv2.imread(path+"/Dogs"+"/"+dog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    count+=1
    #labels.append(1)
    #print(count)

animals=np.array(data)
labels=np.array(labels)

np.save(path+"/animals",animals)
np.save(path+"/labels",labels)