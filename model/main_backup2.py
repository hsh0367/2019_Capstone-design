<<<<<<< HEAD
import keras
import matplotlib
=======
import matplotlib.pyplot as plt
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
<<<<<<< HEAD
from PIL import Image

'''
    - Keras base image classification 
    - Latest update : 10.31
    - Last acc : 80.12%
    - version : 1.52
=======
import tensorflow as tf
from PIL import Image, ImageOps

'''
    backup : 1106 model 2.13 
    ---------- Model Configuration ----------
    |- Keras base image classification      |
    |- Latest update : 11.04                |
    |- LTS model : v1.53                    |
    |- LTS model acc : 80.54%               |
    |- Latest model : v2.12                 |
    |- Latest model acc : 69%               |
    -----------------------------------------
    |- resize input (48,48,1)               |
    |- Performance goal : Accuracy 85% over |
    |- now model : v2.13                    |
    |- label : 100                          | 
    |- Update : 11.05                       | 
    -----------------------------------------
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696

'''


def image_set():
    print("Get images....")
    # image_set()

<<<<<<< HEAD
    batchsize = 64
    image_size = (255, 255)
=======
    batchsize = 32  # ---#
    image_size = (48, 48)
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696

    #
    train_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set/train/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size,
        color_mode='grayscale'
    )

    test_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set/test/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size,
        color_mode='grayscale'
    )
    valid_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/fix_image_set/valid/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size,
        color_mode='grayscale'
    )

    return train_gen, test_gen, valid_gen


def createModel(numclass):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.25))

<<<<<<< HEAD
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
=======
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(Dropout(0.25))  # 0.25 -> 0.1
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(numclass, activation='softmax'))  # label count = dense label.

    return model


<<<<<<< HEAD
import matplotlib.pyplot as plt


def plt_show_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


def plt_show_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


def main():
    print("Start....")

    train_gen, test_gen, valid_gen = image_set()
    numclass = train_gen.num_classes
    print(train_gen.num_classes)
    print("(1) Train model | (2) Load saved model")
    number = input()

    if (number == "1"):
=======
def recommand(predictions, class_dict):
    # Recommend top 3 -> 10
    predictions = predictions[0].tolist()
    predict = []
    for i in range(len(predictions)):
        predict.insert(i, [i, predictions[i]])

    predict2 = sorted(predict, key=lambda x: x[1], reverse=True)

    # recommend1 = [name for name, target in class_dict.items() if target == predict2[0][0]]
    # recommend2 = [name for name, target in class_dict.items() if target == predict2[1][0]]
    # recommend3 = [name for name, target in class_dict.items() if target == predict2[2][0]]
    # print(recommend1, "/", recommend2, "/", recommend3)

    print('== recommand top 10 ==')
    for i in range(10):
        recommand = [name for name, target in class_dict.items() if target == predict2[i][0]]
        recommand_percent = predict2[i][1]
        print(recommand, " ", round(recommand_percent, 3) * 100, "%")


def plt_show_loss(history, name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.savefig('./Result_graph/{}_loss.png'.format(name))


def plt_show_acc(history, name):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.savefig('./Result_graph/{}_acc.png'.format(name))


def save_label_dict(class_label):
    f = open("./save_model/label_dict.txt", "w")
    f.write(str(class_label))
    f.close()


def main():
    # image generate
    print("Start image generate....")
    train_gen, test_gen, valid_gen = image_set()
    numclass = train_gen.num_classes
    class_dict = train_gen.class_indices
    print("label count : ", numclass)

    # save label_dict
    save_label_dict(class_dict)

    print("(1) Train model | (2) Load saved model |")
    number = input()

    if (number == "1"):

>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696
        print("Model build....")
        model1 = createModel(numclass)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Training....")
<<<<<<< HEAD
        history = model1.fit_generator(train_gen, steps_per_epoch=2000, epochs=100, validation_data=valid_gen,
                                       validation_steps=100)
        # origin : step 200 epoch 100
=======
        history = model1.fit_generator(train_gen, steps_per_epoch=5000, epochs=200, validation_data=valid_gen,
                                       validation_steps=100)
        # 10000 / 200 next 30000 100 or 5000 / 300
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696

        # Saving model
        # Save sturcture to json file, weight to h5 file.

        print("Input model's name")
        name = input()
        model1.save('save_model/{}.h5'.format(name))
<<<<<<< HEAD

        plt_show_loss(history)
        plt.show()

        plt_show_acc(history)
        plt.show()

=======

        plt_show_loss(history, name)
        plt.show()
        plt_show_acc(history, name)
        plt.show()

        print("Test....")
        scores = model1.evaluate_generator(test_gen, steps=100)
        print("%s: %.2f%%" % (model1.metrics_names[1], scores[1] * 100))
        print("Test loss : ", scores[0], "/ acc : ", scores[1])
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696

    elif (number == "2"):
        from keras.models import load_model

        model_list = os.listdir('./save_model')
        print('---- Saved model list ----')
        for model_file in model_list:
            print(model_file)

        print("Input model_structure 's name (json only)")
        name = input()
        model1 = load_model('./save_model/{}'.format(name))

        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # model compile


    else:
        print("wrong input")
        exit()

<<<<<<< HEAD
    print("Test....")
    scores = model1.evaluate_generator(test_gen, steps=10)
    print("%s: %.2f%%" % (model1.metrics_names[1], scores[1] * 100))
    print("loss : ", scores[0], "/ acc : ", scores[1])

    model1.summary()

    # ----------------------#
    # Recommended labels top3

    # Print test prediction
    print("-- Predict --")
    batchsize = 64
    image_size = (255, 255)
    pred_gen = ImageDataGenerator().flow_from_directory(
        '/home/mll/Capstone/predict_image/',
        class_mode='categorical',
        batch_size=batchsize,
        target_size=image_size
    )
    predictions = model1.predict_generator(pred_gen)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    import operator
    index, value = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    pred_result = [name for name, target in test_gen.class_indices.items() if target == index]
    print("label : " + index, "| acc : ", value)
    print("-- pred is : " + pred_result)
=======
    model1.summary()

    # import tensorflow as tfjs
    # tfjs.converters.save_keras_model(model1, './save_model')

    # ----------------------#
    # Recommended labels top3

    # Print test prediction
    print("-- Predict --")

    f2 = open('./save_model/label_dict.txt')
    label_dict = eval(f2.read())
    f2.close()

    # pred  -> not imagedatagenerator.
    image_size = (48, 48)

    img = Image.open('/home/mll/Capstone/predict_image/pred/asd.png')
    img = img.resize(image_size, Image.ANTIALIAS)
    img = img.convert('L')  # L
    inv_image = ImageOps.invert(img)

    # show image
    inv_image.show()

    # pred_gen = inv_image.reshape(48, 48, 1)
    pred_gen = np.expand_dims(inv_image, axis=0)
    pred_gen = pred_gen.reshape(1, 48, 48, 1)
    predictions = model1.predict(pred_gen)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    import operator
    index, value = max(enumerate(predictions[0]), key=operator.itemgetter(1))
    pred_result = [name for name, target in label_dict.items() if target == index]
    print("-- pred label : ", index, "| acc : ", value)
    print("-- pred is : \n", pred_result)

    # recommand_top 10
    recommand(predictions, label_dict)

    print(" END! ")
>>>>>>> 44ea940ee6e49f354735fe9e224c511e7e5aa696


if __name__ == "__main__":
    main()
