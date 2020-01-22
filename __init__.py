import numpy
mask = numpy.arange(12).reshape(2, 2, 3)

for x, y in numpy.ndindex(mask.shape[0], mask.shape[1]):
    red = mask[x][y][0]
    green = mask[x][y][1]
    blue = mask[x][y][2]
    print(red, green, blue)


print(numpy.sum(mask))