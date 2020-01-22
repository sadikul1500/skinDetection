import numpy as np
import os
import sys
import cv2
from os.path import join
np.set_printoptions(threshold=sys.maxsize)
np.seterr(divide='ignore', invalid='ignore')



def calculate_probability(skin, non_skin):
    skin_count = np.sum(skin)
    non_skin_count = np.sum(non_skin)
    skin /= skin_count
    non_skin /= non_skin_count
    skin_probability = np.zeros((256, 256, 256))

    for i in range(256):
        for j in range(256):
            for k in range(256):
                if non_skin[i][j][k] == 0.0 and skin[i][j][k] == 0.0 :
                    skin_probability[i][j][k] = 0.0
                elif non_skin[i][j][k] == 0.0 and skin[i][j][k] != 0.0 :
                    skin_probability[i][j][k] = skin[i][j][k]
                else:
                    skin_probability[i][j][k] = skin[i][j][k] / non_skin[i][j][k]
    #skin_probability = p * np.divide(skin, np.add(skin, non_skin))

    return skin_probability



def compareSkin_nonSkin(image , mask, skin, non_skin):

    for x, y in np.ndindex(mask.shape[0], mask.shape[1]):
        red = image[x][y][0]
        green = image[x][y][1]
        blue = image[x][y][2]

        if(mask[x][y][0] >230 and mask[x][y][1] > 230 and mask[x][y][2] > 230):
            non_skin[red][green][blue] += 1
        else:
            skin[red][green][blue] += 1

    return skin, non_skin


def readImage(real_image, mask_image, skin, non_skin):

    for x, y in zip(real_image, mask_image):
        print(x, y)
        image = cv2.imread(join(path_real_image, x))
        mask = cv2.imread(join(path_mask_image, y))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        skin, non_skin = compareSkin_nonSkin(image, mask, skin, non_skin)
    return skin, non_skin


def probability_file(skin_probability):
    out = open('probability.txt', 'w')
    hmm = open('value.txt', 'w')
    #out.write("Red->Green->Blue->Probability\n")

    for i in range(256):
        for j in range(256):
            for k in range(256):
                out.write(str(i)+'->'+str(j)+'->'+str(k)+'->'+str(skin_probability[i][j][k])+'\n')
                if skin_probability[i][j][k] > 0.0:
                    hmm.write(str(i) + '->' + str(j) + '->' + str(k) + '->' + str(skin_probability[i][j][k]) + '\n')

    out.close()
    hmm.close()


if __name__ == '__main__':
    path_real_image = 'G:\\5 th semester\\dbms2\\image'
    real_image = os.listdir(path_real_image)

    path_mask_image = 'G:\\5 th semester\\dbms2\\mask'
    mask_image = os.listdir(path_mask_image)

    skin = np.zeros((256, 256, 256))
    non_skin = np.zeros((256, 256, 256))


    skin, non_skin = readImage(real_image, mask_image, skin, non_skin)
    skin_probability = calculate_probability(skin, non_skin)
    probability_file(skin_probability)
    print('sesh????')
    #print(np.sum(skin_probability))
    cv2.waitKey(0)
    cv2.destroyAllWindows()














'''
import numpy as np
import os
import sys
import cv2
from os.path import join
np.set_printoptions(threshold=sys.maxsize)
np.seterr(divide='ignore', invalid='ignore')

path_real_image = 'G:\\f drive\\1IIT\\5th semester\\dbms2\\test_image'
real_image = os.listdir(path_real_image)

path_mask_image = 'G:\\f drive\\1IIT\\5th semester\\dbms2\\test_mask'
mask_image = os.listdir(path_mask_image)

skin = np.zeros((256, 256, 256))
non_skin = np.zeros((256, 256, 256))
skin_probability = np.zeros((256, 256, 256))

def calculate_probability():
    skin_probability = np.divide(skin, np.add(skin, non_skin))



def compareSkin_nonSkin(image , mask):
    for x, y in np.ndindex(mask.shape[0], mask.shape[1]):
        red = image[x][y][0]
        green = image[x][y][1]
        blue = image[x][y][2]

        if(mask[x][y][0] < 225 and mask[x][y][1] < 225 and mask[x][y][2] < 225):
            skin[red][green][blue] += 1
        else:
            non_skin[red][green][blue] += 1


def readImage():
    for x, y in zip(real_image, mask_image):
        image = cv2.imread(join(path_real_image, x))
        mask = cv2.imread(join(path_mask_image, y))

        compareSkin_nonSkin(image, mask)


def probability_file():
    out = open('probability.txt', 'a')
    out.write("Red->Green->Blue->Probability\n")

    for i in range(256):
        for j in range(256):
            for k in range(256):
                out.write(str(i)+'->'+str(j)+'->'+str(k)+'->'+str(skin_probability[i][j][k])+'\n')

    out.close()


if __name__ == '__main__':
    readImage()
    calculate_probability()
    probability_file()
    print('sesh')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
'''for x, y in zip(glob.glob('G:\\f drive\\1IIT\\5th semester\\dbms2\\image'), glob.glob('G:\\f drive\\1IIT\\5th semester\\dbms2\\image')):
    i = cv2.imread(join(real_image, x))
    print(i)
    j = cv2.imread(y)
    print(j)'''
'''
'''
'''
p = cv2.imread('G:\\f drive\\1IIT\\5th semester\\dbms2\\test_mask\\0000.bmp')
f = open('out.txt', 'a')
#file.write()
print(p, file = f)
#print(p)
print(p.shape)
print(len(p))
f.close()
#cv2.imshow('image', p)

z = np.zeros_like(p)
print(z.shape)

for x, y in np.ndindex(p.shape[0], p.shape[1]):
    if(p[x][y][0] < 220 and p[x][y][1] < 220 and p[x][y][2] < 220):
        z[x][y][0] = 255
        z[x][y][1] = 255
        z[x][y][2] = 255

    else:
        z[x][y][0] = 0
        z[x][y][1] = 0
        z[x][y][2] = 0

'''
#cv2.imshow('image', z)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



'''
for i in range(len(real_image)):
    print(real_image[i], end=' ')
    print(mask_image[i])
'''

'''
'''