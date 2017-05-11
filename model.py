import csv
import cv2
import numpy as np
import os
import sklearn

# implement the generator
samples = []
with open('emdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

print(len(samples))


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_value = 32

def generator(samples, batch_size=batch_value):
    num_samples = len(samples)
    correction = 0.30
    path = 'emdata/IMG/'
    # loop forever so generator never terminates
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            streering_angles = []
            for row in batch_samples:

                steering_center = float(row[3])
                streering_left = steering_center + correction
                streering_right = steering_center - correction
                streering_center_flip = steering_center * -1.0

                img_center = np.asarray(cv2.imread(path + row[0].split('/')[-1]))
                img_center_flip = np.asarray(cv2.flip(cv2.imread(path + row[0].split('/')[-1]),1))
                img_left = np.asarray(cv2.imread(path + row[1].split('/')[-1]))
                img_right = np.asarray(cv2.imread(path + row[2].split('/')[-1]))

                car_images.extend([img_center, img_center_flip, img_left, img_right])
                streering_angles.extend([steering_center, streering_center_flip, streering_left, streering_right])

            X_train = np.array(car_images)
            y_train = np.array(streering_angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_value)
validation_generator = generator(validation_samples, batch_size=batch_value)

'''
lines = []
with open('emdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'emdata/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    # removed the header row to avoid string to float error
    measurements.append(measurement)

# Double the images by adding flipped centered images
augmented_images, augmented_measures = [], []
for image, measurement in zip(images, measurements):dfdfdsf
    augmented_images.append(image)
    augmented_measures.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measures.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measures)



# work ont the 3 camera, the run only 1 epoche, took 30 minutes 0.17 accuracy, stable at the beginning but failed in a turn with no right side track.
# comment at this time to use the flipped center image for now.  Will revisit this portion later.  May combined to augmented data soon.
car_images = []
streering_angles = []
with open('emdata/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        correction = 0.30
        streering_left = steering_center + correction
        streering_right = steering_center - correction
        streering_center_flip = steering_center * -1.0

        path = 'emdata/IMG/'
        img_center = np.asarray(cv2.imread(path + row[0].split('/')[-1]))
        img_center_flip = np.asarray(cv2.flip(cv2.imread(path + row[0].split('/')[-1]),1))
        img_left = np.asarray(cv2.imread(path + row[1].split('/')[-1]))
        img_right = np.asarray(cv2.imread(path + row[2].split('/')[-1]))

        car_images.extend([img_center, img_center_flip, img_left, img_right])
        streering_angles.extend([steering_center, streering_center_flip, streering_left, streering_right])

X_train = np.array(car_images)
y_train = np.array(streering_angles)


'''

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D


# Use LeNet model type by adding 2 Conv type and 2 Pooling
# Later on, change to Nidiva method but add a dropout to prevent overfitting
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Convolution2D(6,5,5,activation="relu"))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Flatten())
# align more center after adding the Dropout, orderwise it is more to the right
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
# Use model.fit for non generator version
# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2, verbose=1)
# Use model.fit_generator for generator version
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples) // batch_value, validation_data=validation_generator, validation_steps=len(validation_samples) // batch_value, epochs=5, verbose=1 )

model.save('model.h5')

import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
