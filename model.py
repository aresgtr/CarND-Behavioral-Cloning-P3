import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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

train_samples, validation_samples = train_test_split(lines, test_size=0.15)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                filename = filename.split('\\')[-1]
                current_path = image_dir + filename
                center_image = cv2.imread(current_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                image_flipped = np.fliplr(center_image)
                angle_flipped = -center_angle
                images.append(image_flipped)
                angles.append(angle_flipped)

                # Left
                source_path = batch_sample[1]
                filename = source_path.split('/')[-1]
                filename = filename.split('\\')[-1]
                current_path = image_dir + filename
                left_image = cv2.imread(current_path)
                left_angle = float(batch_sample[3]) + 0.2
                images.append(left_image)
                angles.append(left_angle)

                # Right
                source_path = batch_sample[2]
                filename = source_path.split('/')[-1]
                filename = filename.split('\\')[-1]
                current_path = image_dir + filename
                right_image = cv2.imread(current_path)
                right_angle = float(batch_sample[3]) - 0.2
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

"""
Neural Network
"""
model = Sequential()

# Normalize Data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# Crop off unused
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))

model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())

# Dropout layer
model.add(Dropout(0.25, noise_shape=None, seed=None))

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')
