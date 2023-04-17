from keras.models import model_from_json
from keras.utils import np_utils
import keras
from PIL import Image
import numpy as np
import os
import cv2

path = os.getcwd()

animals=np.load(path+"/animals.npy")
labels=np.load(path+"/labels.npy")
s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]
num_classes=len(np.unique(labels))
data_length=len(animals)

(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

json_file = open(path+'/Models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(path+"/Models/model.h5")
print("Loaded model from disk")

# compile the model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
#use Categorical Cross Entropy for more than 1 labels

score = loaded_model.evaluate(x_test, y_test, verbose=1)
print('/n', 'Test accuracy:', score[1])

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def get_animal_name(label):
    if label==1:
        return "dog"
    if label==0:
        return "cat"

def predict_single_animal_GUI():
    pass    


def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=loaded_model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a "+animal+" with accuracy = "+str(acc))
