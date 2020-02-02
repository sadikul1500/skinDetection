import numpy
mask = numpy.arange(12).reshape(2, 2, 3)

for x, y in numpy.ndindex(mask.shape[0], mask.shape[1]):
    red = mask[x][y][0]
    green = mask[x][y][1]
    blue = mask[x][y][2]
    print(red, green, blue)


print(numpy.sum(mask))



import os
from os.path import join
from random import shuffle

path_real_image = 'G:\\5 th semester\\dbms2\\image'
real_image = os.listdir(path_real_image)
print(type(real_image))


path_mask_image = 'G:\\5 th semester\\dbms2\\mask'
mask_image = os.listdir(path_mask_image)
c = list(zip(real_image, mask_image))



shuffle(c)

a, b = zip(*c)

for x, y in zip(a,b):
    print(x, y)
