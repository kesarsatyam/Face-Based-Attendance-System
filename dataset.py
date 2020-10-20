import cv2
import numpy as np
def main(name):
    # Detect object in video stream using Haarcascade Frontal Face
    face_casecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    def face_extractor(img):
        # Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect frames of different sizes, list of faces rectangles
        faces = face_casecade.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return None
        # Loops for each faces
        for (x, y, w, h) in faces:
            # Crop the image frame into rectangle
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    # Start capturing video
    cap = cv2.VideoCapture(0)
    count = 0 # Initialize variable for sample face image
    # Start looping
    while True:
        # Capture video frame
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(400,400))
            # Convert frame to grayscale
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

            file_name_path = 'dataset/'+name+'/'+str(count)+'.jpg'
            # Save the captured image into the datasets folder
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('ATTENDANCE',face)
        else:
            print("Face not found")
            pass
        # To stop taking video, press 'q' for at least 80ms
        if cv2.waitKey(1)==13 or count==80:
        #if count == 50:
            break
    cap.release() # Stop video
    cv2.destroyAllWindows() # Close all started windows
    print('Samples Collection Completed')
# name=input()
#main('abc')
#name = 'shivam'
#main('satyam')
#main('shivam')
main('SATYAMM')