import keras
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

def image_set():
    image_path = '/home/mll/Capstone/fix_image_set'

    folder = os.listdir(image_path)
    folder.sort()
    label_list = []

    for data_folder in folder:
        data_folder_path = os.path.join(image_path, '{}'.format(data_folder))
        data_folder_list = os.listdir(data_folder_path)
        data_folder_list.sort()
        label_list.append(data_folder)

        for image in data_folder:
            print("")


def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(255,255,3)))
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
    model.add(Dense(512, activation='softmax'))
    model.add(Dropout(0.5))
    return model


def main():
    print("Start....")

    print("Get images....")
    # image_set()

    batchsize = 32
    image_size = [255, 255]

    #
    train_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set'+,
        class_mode='categorical',
        batchsize = batchsize,
        target_size=(255, 255),
    )

    test_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set',
        class_mode='categorical',
        batchsize = batchsize,
        target_size=(255, 255),
    )

    vaild_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set',
        class_mode='categorical',
        batchsize=batchsize,
        target_size=(255, 255),
    )


    print("Model build....")
    model1 = createModel()

    #model 가시화
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))

    model1.summary()
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    print("Training....")
    model1.fit_generator(train_gen, steps_per_epoch=100, epochs=100, validation_data=vaild_gen, vaild_gen=5)
    model.save_weights('save_model.h5')
    print("Test....")
    scores = model1.evaluate_generator(test_gen, steps=5)
    print("%.2f%%" %(scores[1],100))



if __name__ == "__main__":
    main()
