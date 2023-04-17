import Predict_Animal
import os

path = os.getcwd()

count = 0
for i in range(1,6):
    print("Currently on Sample "+str(i))
    Predict_Animal.predict_animal(path+"/test1/"+str(i)+".jpg")
