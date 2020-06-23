import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

"""
Import training data
"""
csv_location = 'D:\\UdacityCarData\\data\\driving_log.csv'
# csv_location = '/opt/carnd_p3/data/driving_log.csv'
image_dir = 'D:\\UdacityCarData\\data\\IMG\\'
# image_dir = '/opt/carnd_p3/data/IMG/'

lines = []
with open(csv_location) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip first line
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = image_dir + filename
    image = cv2.imread(current_path)
    image_flipped = np.fliplr(image)
    images.append(image)
    images.append(image_flipped)
    measurement = float(line[3])
    measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement_flipped)

X_train_0 = np.array(images)
y_train_0 = np.array(measurements)

# For my own data
csv_location = 'D:\\UdacityCarData\\my_own_data\\driving_log.csv'
# csv_location = '/opt/carnd_p3/data/driving_log.csv'
image_dir = 'D:\\UdacityCarData\\my_own_data\\IMG\\'
# image_dir = '/opt/carnd_p3/data/IMG/'

lines = []
with open(csv_location) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip first line
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    filename = filename.split('\\')[-1]
    current_path = image_dir + filename
    image = cv2.imread(current_path)
    image_flipped = np.fliplr(image)
    images.append(image)
    images.append(image_flipped)
    measurement = float(line[3])
    measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement_flipped)

X_train_1 = np.array(images)
y_train_1 = np.array(measurements)

X_train = np.append(X_train_0, X_train_1)
y_train = np.append(y_train_0, y_train_1)

print(len(y_train))
"""
Neural Network
"""
model = Sequential()

# Normalize Data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Crop off unused
model.add(Cropping2D(cropping=((70, 25), (10, 10))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))

model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

# Dropout layer
model.add(Dropout(0.5, noise_shape=None, seed=None))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
