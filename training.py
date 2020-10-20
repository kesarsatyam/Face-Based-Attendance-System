import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


def main(n):
    c = 0
    # get the path of all the files in the folder
    data_path = 'dataset/' + n + '/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # two emptly list for training dataset and for the labels of the image
    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        # loading the image and converting it to gray scale
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Now we are converting the image into numpy array and appebding to training dataset
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)  # and Labels

    Labels = np.asarray(Labels, dtype=np.int32)
    # Local Binary Pattern (LBP) is a simple yet very efficient texture operator
    # which labels the pixels of an image by thresholding the neighborhood of each pixel
    # and considers the result as a binary number.

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Completed")
#main('satyam')
#main('SATYAMM')