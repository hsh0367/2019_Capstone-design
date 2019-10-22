import tensorflow as tf
import numpy as np
import os.path as path
from scipy import misc

def image_set():
    image_path = ''
    images = [misc.imread(path) for path in image_path]
    images = np.asarray(images)
    image_size = np.asarray([images.shape[1], images.shape[2],images.shape[3]])
    print(image_size)

def model(X):
    print("model....")


def main():
    print("Start....")




if __name__ == "__main__":
    main()
