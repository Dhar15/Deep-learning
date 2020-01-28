#IMAGE RECOGNITION AND CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS

#Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64,64,3), activation = 'relu'))
#32 is the no.of feature maps of 3*3 dimension. The input shape is the shape
#of the input image, it has a resolution of 64*64 and the 3 indicates it is 
#coloured. For B/W it would be 1. 

#Step 2 - Max Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolution and max pooling layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection (Making a classic ANN. The flattened result 
#                          is the input layer to the ANN)
classifier.add(Dense(activation='relu', units=128)) #Hidden layer
classifier.add(Dense(activation = 'sigmoid', units=1)) #Output layer

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator #Used for data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch=8000,
                         epochs = 25, validation_data = test_set, 
                         validation_steps = 2000)
