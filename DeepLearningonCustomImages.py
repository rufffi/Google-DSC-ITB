import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings('ignore')

def displayImg(img,cmap=None):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Keras has a built in function that generates batches from a directory, it's called the image data generator
from keras.preprocessing.image import ImageDataGenerator
#To create a more robust trainer, we generate different random orientations for the data
image_gen = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=False,fill_mode='nearest')

#Creating model
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
model = Sequential()

#Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
#Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Convolutional Layer
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
#Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Convolutional Layer
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
#Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Flatten Layer
model.add(Flatten())
#Dense Layer
model.add(Dense(128,activation='relu'))
#Dropout Layer (Helps reduce overfitting, randomly turn off neurons e.g 50% of neurons)
model.add(Dropout(0.5))
#Output Layer
model.add(Dense(1,activation='sigmoid'))
#Compilation
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Summary
print(model.summary())

#To start the training you need to choose the batch size (just starting point)
batch_size = 16
#Generate manipulated images from the directory
train_image_gen = image_gen.flow_from_directory('C:/PythonPrograms/Course/OpenCV Course/DEEP LEARNING/CATS_DOGS/train',target_size=(150,150),batch_size=batch_size,class_mode='binary')
test_image_gen = image_gen.flow_from_directory('C:/PythonPrograms/Course/OpenCV Course/DEEP LEARNING/CATS_DOGS/test',target_size=(150,150),batch_size=batch_size,class_mode='binary')
print(train_image_gen.class_indices)

results = model.fit_generator(train_image_gen,epochs=10,steps_per_epoch=150,validation_data=test_image_gen,validation_steps=12)

#Saving model
model.save('C:\PythonPrograms\Deep Learning Models\10epochscatdog.h5')