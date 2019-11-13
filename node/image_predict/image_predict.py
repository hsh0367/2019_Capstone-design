import matplotlib
import numpy as np
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import operator
from keras.models import load_model
from keras.preprocessing import image
import base64
import ast
import sys
from io import BytesIO
import timeit

def recommand(predictions, class_dict):
    # Recommend top 3
    predictions = predictions[0].tolist()
    predict = []
    for i in range(len(predictions)):
        predict.insert(i, [i, predictions[i]])

    predict2 = sorted(predict, key=lambda x: x[1], reverse=True)

    re_str = ""
    for i in range(10):
        recommend = [name for name, target in class_dict.items() if target == predict2[i][0]]
        re_str = re_str + str(recommend)[2:len(recommend)-3]
        if i == 9:
            break
        re_str += ","
    return re_str

'''
    recommend1 = str([name for name, target in class_dict.items() if target == predict2[0][0]])
    recommend2 = str([name for name, target in class_dict.items() if target == predict2[1][0]])
    recommend3 = str([name for name, target in class_dict.items() if target == predict2[2][0]])
    re_str = recommend1[2:len(recommend1)-2]+","+recommend2[2:len(recommend2)-2]+","+recommend3[2:len(recommend3)-2]
    return re_str
'''

image_load_start = timeit.default_timer()

# base64.txt -> image -> save folder
remove_str = 'data:image/png;base64,'

# image_path = '/Users/gangsin-won/2019_Capstone-design/node/image_predict/predict_image/12/out.png'
# img = open(image_path, 'wb')
# img.write(base64.b64decode(sys.argv[1][len(remove_str):]))
img = Image.open(BytesIO(base64.b64decode(sys.argv[1][len(remove_str):])))
img = img.resize((255, 255))
img = img.convert("RGB")


image_load_stop = timeit.default_timer()

pred_gen = np.expand_dims(img, axis=0)

model_load_start = timeit.default_timer()

model1 = load_model('/Users/gangsin-won/2019_Capstone-design/node/image_predict/model.h5')
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Print test prediction

f2 = open('/Users/gangsin-won/2019_Capstone-design/node/image_predict//label_dict.txt')
label_dict = eval(f2.read())
f2.close()

model_load_stop = timeit.default_timer()
pre_time_start = timeit.default_timer()

predictions = model1.predict(pred_gen)

pre_time_stop = timeit.default_timer()
else_time_start = timeit.default_timer()

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
index, value = max(enumerate(predictions[0]), key=operator.itemgetter(1))
pred_result = [name for name, target in label_dict.items() if target == index]

#recommand_top3
re_list = recommand(predictions, label_dict)

else_time_stop = timeit.default_timer()

print('image_load: ', image_load_stop - image_load_start)
print('model_load: ', model_load_stop - model_load_start)
print('pre_time: ', pre_time_stop - pre_time_start)
print('else_time: ', else_time_stop - else_time_start)
print(re_list)
