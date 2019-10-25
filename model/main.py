import keras
import matplotlib
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

def image_set():
    print("Get images....")
    # image_set()

    batchsize = 64
    image_size = (255, 255)

    #
    train_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set/train/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size
    )

    test_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set/test/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size
    )
    valid_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set/valid/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size
    )

    return train_gen, test_gen, valid_gen


def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='sigmoid', input_shape=(255, 255, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='softmax'))  # label count = dense label.

    return model



def main():
    print("Start....")

    train_gen, test_gen, valid_gen = image_set()
    batchsize = 64

    #model 가시화
    #from IPython.display import SVG
    #from keras.utils.vis_utils import model_to_dot
    #SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))



    print("(1) Train model | (2) Load saved model")
    number = input()

    if(number=="1"):
        print("Model build....")
        model1 = createModel()
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model1.summary()
        '''
        from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot
        import matplotlib
        SVG(model_to_dot(model1, show_shapes=True).create(prog='dot', format='svg'))
        '''

        print("Training....")
        model1.fit_generator(train_gen, steps_per_epoch=50, epochs=20, validation_data=valid_gen, validation_steps=10)

        #Saving model
        #Save sturcture to json file, weight to h5 file.

        print("Input model's name")
        name = input()
        model1.save_weights('save_model/{}_model.h5'.format(name))
        '''
        model_json = model1.to_json()
        with open('save_model_structure/{}.json'.format(name), "w") as json_file:
            json_file.write(model_json)
        '''


    elif(number=="2"):
        from keras.models import load_model


        model_list = os.listdir('./save_model')
        print('---- Saved model list ----')
        for model_file in model_list:
            print(model_file)

        print("Input model_structure 's name (json only)")
        name = input()
        model1 = load_model('save_model/{}'.format(name))
        '''
        from keras.models import model_from_json 
        json_file = open('save_model/{}'.format(name), "r")
        loaded_model_json = json_file.read()
        json_file.close() 
        loaded_model = model_from_json(loaded_model_json)
        '''
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # model compile


    else:
        print("wrong input")
        exit()

    print("Test....")
    scores = model1.evaluate_generator(test_gen, steps=50)

    print((scores,100))


    #recommended labels top3




if __name__ == "__main__":
    main()
