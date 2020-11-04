import cv2
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime
now = datetime.now()
def main(n):
    c=0
    # get the path of all the files in the folder
    data_path = 'dataset/' + n + '/'
    onlyfiles = [ f for f in listdir(data_path) if isfile(join(data_path,f))]
    # two emptly list for training dataset and for the labels of the image
    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        # loading the image and converting it to gray scale
        images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        # Now we are converting the image into numpy array and appebding to training dataset
        Training_Data.append(np.asarray(images,dtype=np.uint8))
        Labels.append(i) # and Labels

    Labels = np.asarray(Labels, dtype=np.int32)
    #Local Binary Pattern (LBP) is a simple yet very efficient texture operator
    #which labels the pixels of an image by thresholding the neighborhood of each pixel
    #and considers the result as a binary number.

    model = cv2.face.LBPHFaceRecognizer_create()


    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Completed")

    # Recognition

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # extract the face from the training image sample
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        # if face is not there then simply return none
        if faces is():
            return img,[]
        #
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0) # strat capturing video
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # convert to gray scale
            result = model.predict(face) # predict the image from tained model

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))

            if confidence > 85: # if confidance is greater than 85 then we will confirm that user.
                cv2.putText(image, n, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # put text on the frame
                cv2.imshow('ATTENDANCE', image)
                c=1 # make true for attendance
            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2) # show unkonwn
                cv2.imshow('ATTENDANCE', image)
        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) # if no face is there
            cv2.imshow('ATTENDANCE', image)
            pass
        if cv2.waitKey(1)==13: # 13 is for enter
            break
    cap.release() # stop video
    cv2.destroyAllWindows() # Close all started windows
    # TO UPDATE USERS ATTENDANCE IN EXCEL FILE
    if c==1:
        pose_x = n
        pose_y = 'P'

        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #print("date and time =", dt_string)
        pose_z = dt_string
        #with open('Demo.csv', mode='a') as file_:
        with open('StudentList.csv', mode='a') as file_:
            file_.write("{},{},{}".format(pose_x, pose_y,pose_z))
            file_.write("\n")
# Further we can make GUI for to where we can list the users details to mark the attendance as well register themsalves.
#n=str(input())
#main('shivam')
#main('satyam')
#main('abc')
main('SATYAMM')
