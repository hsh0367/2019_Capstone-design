import struct
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import pylab


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


for drawing in unpack_drawings('airplane.bin'):
    image_array = drawing['image']
    image = np.zeros((256, 256))

    for x in range(len(image_array)):
        x1, y1 = image_array[x][0], image_array[x][1]
        plt.plot(x1, y1, color='k')
        plt.axis('off')
        pylab.savefig('mypic.png', bbox_inches='tight', pad_inches=0)
