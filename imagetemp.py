import csv
import cv2
from matplotlib import pyplot as plt

"""
Import training data
"""
csv_location = 'D:\\UdacityCarData\\data\\driving_log.csv'
image_dir = 'D:\\UdacityCarData\\data\\IMG\\'

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
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    plt.imshow(image)
    plt.show()
    break