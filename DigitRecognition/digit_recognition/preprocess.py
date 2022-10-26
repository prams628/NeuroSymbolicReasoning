# import the necessary libraries
import os

import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Setting global variables and seed
SEED_VAL = 42
np.random.seed(SEED_VAL)
PATH = '../data/'
IMAGE_LENGTH = 64
IMAGE_WIDTH = 64

# init empty arrays
X = np.empty((0, 1, IMAGE_LENGTH, IMAGE_WIDTH))
y = np.array([])
labels = os.listdir(PATH)

for label in labels:
    print(f'Processing {label}')
    individual_files = os.listdir(os.path.join(PATH, label))
    label_array = np.empty((0, 1, IMAGE_LENGTH, IMAGE_WIDTH))
    for _file in individual_files:
        # read the image in grayscale mode
        img = cv2.imread(os.path.join(PATH, label, _file), 0)

        # normalise the data between 0 and 1
        img = img / 255

        label_array = np.append(label_array, np.reshape(img, (1, 1, IMAGE_LENGTH, IMAGE_WIDTH)), axis=0)

    X = np.append(X, label_array, axis=0)
    y = np.append(y, np.array([int(label) for _ in range(len(individual_files))]))

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(np.reshape(y, (-1, 1)).astype(np.uint8))

# convert variable to uint8 to occupy less memory
y_ohe = y_ohe.astype(np.uint8)

# split data into train, test, and val in the ratio 72-20-8
X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.2, shuffle=True, random_state=SEED_VAL)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=SEED_VAL)

print('Data has been generated. Saving data...')
np.save('X_digit_train', X_train)
np.save('y_digit_train', y_train)
np.save('X_digit_val', X_val)
np.save('y_digit_val', y_val)
np.save('X_digit_test', X_test)
np.save('y_digit_test', y_test)
print('Done.')
