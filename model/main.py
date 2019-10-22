import keras
import numpy as np
import os.path as path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


def image_set():
    image_path = ''
    images = [misc.imread(path) for path in image_path]
    images = np.asarray(images)
    image_size = np.asarray([images.shape[1], images.shape[2],images.shape[3]])
    print(image_size)



def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    return model

def main():
    print("Start....")

    train_data = ()

    print("Model build....")
    model1 = createModel()

    print("Train start....")


    model1 = createModel()





if __name__ == "__main__":
    main()
