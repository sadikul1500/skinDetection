#from skin_detection import skinTraining
import numpy as np
import os
#import sys
import cv2
from os.path import join

#print(skinTraining.skin_probability)

def test(image, probability):
    img = image
    i = 0
    for x, y in np.ndindex(image.shape[0], image.shape[1]):
        red = image[x][y][0]
        green = image[x][y][1]
        blue = image[x][y][2]

        if probability[red][green][blue] < .48:
            img[x][y][0] = 255
            img[x][y][1] = 255
            img[x][y][2] = 255


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    #cv2.imwrite('G:\\5 th semester\\dbms2\\nota', img)
    path = 'G:\\5 th semester\\dbms2\\result'
    cv2.imwrite(os.path.join(path, str(i)+'.jpg'), img)
    i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def readImage(path_test_image, test_image, probability):

    for x in (test_image):
        image = cv2.imread(join(path_test_image, x))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        test(image, probability)

if __name__ == "__main__":

    path_test_image = 'G:\\5 th semester\\dbms2\\test'
    test_image = os.listdir(path_test_image)

    probability = np.zeros((256, 256, 256))
    file = open('probability.txt', 'r')
    lines = file.readlines()

    for line in lines:
        r, g, b, prob = line.split('->')
        probability[int(r)][int(g)][int(b)] = float(prob)

    readImage(path_test_image, test_image, probability)


