import numpy as np
import os
import sys
import cv2
from os.path import join
from random import shuffle
np.set_printoptions(threshold=sys.maxsize)
np.seterr(divide='ignore', invalid='ignore')


index = 0

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

        image = cv2.resize(image, (100, 90), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (100, 90), interpolation=cv2.INTER_AREA)

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


def test(image, mask, probability):
    img = image
    global index
    count = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0

    for x, y in np.ndindex(image.shape[0], image.shape[1]):
        red = image[x][y][0]
        green = image[x][y][1]
        blue = image[x][y][2]

        if probability[red][green][blue] < .58:
            img[x][y][0] = 255
            img[x][y][1] = 255
            img[x][y][2] = 255

            if mask[x][y][0] == 255 and mask[x][y][1] == 255 and mask[x][y][2] == 255:
                count += 1
                true_negative += 1
            else:
                false_negative += 1

        elif (mask[x][y][0] != 255 and mask[x][y][1] != 255 and mask[x][y][2] != 255) and (img[x][y][0] != 255 and img[x][y][1] != 255 and img[x][y][2] != 255):
            count += 1
            true_positive += 1
        elif (mask[x][y][0] == 255 and mask[x][y][1] == 255 and mask[x][y][2] == 255) and (img[x][y][0] != 255 and img[x][y][1] != 255 and img[x][y][2] != 255):
            false_positive += 1

    epsilon = 1e-10
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    accuracy = count / (image.shape[0] * image.shape[1])
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    # cv2.imwrite('G:\\5 th semester\\dbms2\\nota', img)
    path = 'G:\\5 th semester\\dbms2\\result'
    cv2.imwrite(os.path.join(path, str(index) + '.jpg'), img)
    cv2.waitKey(0)
    index += 1

    return accuracy, precision, recall
    #cv2.destroyAllWindows()


def testImages(test_real_image, test_mask_image, probability):
    output = []
    precision = []
    recall = []
    #print(test_real_image)
    for x, y in zip(test_real_image, test_mask_image):
        image = cv2.imread(join(path_real_image, x))
        mask = cv2.imread(join(path_mask_image, y))

        width = int(image.shape[0] * .7)
        height = int(image.shape[0] * .8)


        dsize = (width, height)

        image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_AREA)
        #print(type(image))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        accuracy, prec, rec = test(image, mask, probability)
        output.append(accuracy)
        precision.append(prec)
        recall.append(rec)

    return output, precision, recall



if __name__ == '__main__':
    path_real_image = 'G:\\5 th semester\\dbms2\\image'
    real_image = os.listdir(path_real_image)

    path_mask_image = 'G:\\5 th semester\\dbms2\\mask'
    mask_image = os.listdir(path_mask_image)
    #print(type(real_image))

    avarage_accuracy = []
    avarage_precision = []
    avarage_recall = []
    avarage_f_measure = []

    k = int(input('enter number of fold: '))

    skin = np.zeros((256, 256, 256))
    non_skin = np.zeros((256, 256, 256))

    combined = list(zip(real_image, mask_image))
    shuffle(combined)
    real_image, mask_image = zip(*combined)

    percent = int(len(real_image) / k)

    for i in range(k):

        #print(type(real_image))
        #split_index = int((1 - k/100) * len(real_image))
        start = i * percent
        end = start + percent
        train_real_image = real_image[:start] + real_image[end-1:]
        train_mask_image = mask_image[:start] + mask_image[end-1:]
        print(train_real_image)
        test_real_image = real_image[start : end]
        test_mask_image = mask_image[start : end]

        test_real_image = [e for e in test_real_image]
        test_mask_image = [e for e in test_mask_image]
        train_real_image = [e for e in train_real_image]
        train_mask_image = [e for e in train_mask_image]

        print(train_real_image)
        skin, non_skin = readImage(train_real_image, train_mask_image, skin, non_skin)
        skin_probability = calculate_probability(skin, non_skin)
        #probability_file(skin_probability)
        print('sesh????')


        accuracy, precision, recall = testImages(test_real_image, test_mask_image, skin_probability)
        #mean =
        avarage_accuracy.append(sum(accuracy) / len(accuracy))
        avarage_precision.append(sum(precision) / len(precision))
        avarage_recall.append(sum(recall) / len(recall))
        f1 = 2 * avarage_precision[-1] * avarage_recall[-1] / (avarage_recall[-1] + avarage_precision[-1])
        avarage_f_measure.append(f1)

        print('accuracy', avarage_accuracy)
        print('precision', avarage_precision)
        print('recall', avarage_recall)
        print('measure', avarage_f_measure)
        #print(mean)
        #print(np.sum(skin_probability))
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

    print('mean accuracy : ', sum(avarage_accuracy) / len(avarage_accuracy))
    print('mean precision : ', sum(avarage_precision) / len(avarage_precision))
    print('mean recall : ', sum(avarage_recall) / len(avarage_recall))
    print('mean f_measure : ', sum(avarage_f_measure) / len(avarage_f_measure))












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