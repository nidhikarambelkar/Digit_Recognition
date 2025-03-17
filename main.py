#Implementing Recognition of Handwritten Digits (MNIST) Datasets with CNN
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#Load MNIST dataset

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
#Check x_train size
x_train.shape
#(60000, 28, 28)- Implies that there are 60000 images with 28 pixels height and width
#Grayscale images

#Normalize the dataset between 0 and 1
x_train=x_train/255.0
x_test=x_test/255.0

#Reshape images to (28,28,1) to match CNN input
#Previous it was (28,28)
#-1 indicates the anything can be the number of rows
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

#Convert labels to one-hot encoding
y_train=keras.utils.to_categorical(y_train,10)#0 to 9 digits
y_test=keras.utils.to_categorical(y_test,10)

#Build the CNN model
model=keras.Sequential(
    [
     keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
     keras.layers.MaxPooling2D((2,2)),
     keras.layers.Conv2D(64,(3,3),activation='relu'),
     keras.layers.MaxPooling2D((2,2)),
     keras.layers.Conv2D(128,(3,3),activation='relu'),
     keras.layers.Flatten(),
     keras.layers.Dense(128,activation='relu'),
     keras.layers.Dropout(0.2),
     keras.layers.Dense(10,activation='softmax')])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_test,y_test))
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Accuracy ",test_acc)

predictions=model.predict(x_test)
print(predictions)

def plot_images(images,labels,predictions,num_samples=5):
    plt.figure(figsize=(10,4))
    for i in range(num_samples):
        plt.subplot(1,num_samples,i+1)
        plt.imshow(images[i].reshape(28,28),cmap='gray')
        predicted_label=np.argmax(predictions[i])
        plt.title(f"Predicted label: {predicted_label}")
        plt.axis('off')
    plt.show()    
plot_images(x_test, y_test, predictions)
