import cv2
import os
from argparse import ArgumentParser

videoNumber = 1

def detect_and_save_face(name, quantity):
    directory = './data/'+name
    if not os.path.exists(directory):
        print "making a new directory: {0}".format(directory)
        os.mkdir(directory)

    face_detector = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(videoNumber)
    count = -24
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,
                                                scaleFactor=1.3,
                                                minNeighbors=5,
                                                minSize=(200, 200),
                                                maxSize=(400, 400))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            if(count>=0):
                filename = str(count) + '.pgm'
                filepath = os.path.join(directory, filename)
                print 'writing {0}'.format(filepath)
                cv2.imwrite(filepath, gray[y:y+h, x:x+w])
            count += 1
        cv2.imshow("camera", frame)
        if count>=quantity or cv2.waitKey(10) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', dest='person_name', type=str,
                        help='the name of the person who is going to be add to database')
    parser.add_argument('-num', '--number', dest='image_number', type=int, default=100,
                        help="how many images of one person's face will be record")
    args = parser.parse_args()
    person_name = args.person_name
    face_quantity = args.image_number

    detect_and_save_face(person_name, face_quantity)
