import cv2
import os
from argparse import ArgumentParser
import time

videoNumber = 1

def get_names(txt_file):
    names = []
    with open(txt_file, mode='r') as f:
        for line in f:
            names.append(line.strip())
    return names

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str, default='LBPH',
                        help='face recognizer model name: LBPH, Eigen, Fisher')
    parser.add_argument('-md', '--model-directory', dest='model_dir', type=str, default='./models',
                        help='a directory where to load the trained model')
    return parser.parse_args()

def Eigen_recognizer(model_dir):
    model_path = os.path.join(model_dir, 'Eigen.yml')
    face_recognizer = cv2.createEigenFaceRecognizer(5)
    print 'loading model in {0}'.format(model_path)
    face_recognizer.load(model_path)
    print 'load model finished.'
    return face_recognizer

def LBPH_recognizer(model_dir):
    model_path = os.path.join(model_dir, 'LBPH.yml')
    face_recognizer = cv2.createLBPHFaceRecognizer()
    print 'loading model in {0}'.format(model_path)
    face_recognizer.load(model_path)
    print 'load model finished.'
    return face_recognizer

def Fisher_recognizer(model_dir):
    model_path = os.path.join(model_dir, 'Fisher.yml')
    face_recognizer = cv2.createFisherFaceRecognizer()
    print 'loading model in {0}'.format(model_path)
    face_recognizer.load(model_path)
    print 'load model finished.'
    return face_recognizer

def Both_recognizer(model_dir):
    face_recognizer_fisher = cv2.createFisherFaceRecognizer()
    print 'loading model in {0}'.format(os.path.join(model_dir, 'Fisher.yml'))
    face_recognizer_fisher.load(os.path.join(model_dir, 'Fisher.yml'))
    print 'load Fisher model finished.'

    face_recognizer_eigen = cv2.createEigenFaceRecognizer(5)
    print 'loading model in {0}'.format(os.path.join(model_dir, 'Eigen.yml'))
    face_recognizer_eigen.load(os.path.join(model_dir, 'Eigen.yml'))
    print 'load Eigen model finished.'

    face_recognizer_lbph = cv2.createLBPHFaceRecognizer()
    print 'loading model in {0}'.format(os.path.join(model_dir, 'LBPH.yml'))
    face_recognizer_lbph.load(os.path.join(model_dir, 'LBPH.yml'))
    print 'load LBPH model finished.'

    return [face_recognizer_lbph, face_recognizer_fisher, face_recognizer_eigen]

def face_predict(model, roi, face_recognizer):
    if isinstance(face_recognizer, list) and model == 'Both':
        label_list, confidence_list = [-1, -1, -1], [-0.9, -0.9, -0.9]

        label_list[0], confidence_list[0] = face_recognizer[0].predict(roi)
        roi = cv2.resize(roi, (360, 360))
        label_list[1], confidence_list[1] = face_recognizer[1].predict(roi)
        label_list[2], confidence_list[2] = face_recognizer[2].predict(roi)

        if label_list[0] == label_list[1]:
            return label_list[0], (confidence_list[0] + confidence_list[1]) / 2
        elif label_list[0] == label_list[2]:
            return label_list[0], (confidence_list[0] + confidence_list[2]) / 2
        elif label_list[1] == label_list[2]:
            return label_list[1], (confidence_list[1] + confidence_list[2]) / 2
        else:
            return -1, -1
    elif model == 'Eigen':
        roi = cv2.resize(roi, (360, 360))
        return face_recognizer.predict(roi)
    elif model == 'Fisher':
        roi = cv2.resize(roi, (360, 360))
        return face_recognizer.predict(roi)
    elif model == 'LBPH':
        return face_recognizer.predict(roi)
    else:
        return -1, -1

def main():
    args = parse_args()
    model = args.model
    model_dir = args.model_dir
    namesfile_path = os.path.join(model_dir, 'names.txt')
    # model_path = os.path.join(model_dir, model+'.yml')

    names = get_names(namesfile_path)
    print "names: " + str(names)

    face_detector = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(videoNumber)

    face_recognizer = None
    if model == 'Eigen':
        face_recognizer = Eigen_recognizer(model_dir)
    elif model == 'Fisher':
        face_recognizer = Fisher_recognizer(model_dir)
    elif model == 'LBPH':
        face_recognizer = LBPH_recognizer(model_dir)
    elif model == 'Both':
        face_recognizer = Both_recognizer(model_dir)
    else:
        raise Exception('model name error!')

    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(200, 200), maxSize=(400, 400))
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x,y), (x+w), (255,0,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]

            predicted_label, predicted_confidence = face_predict(model, roi, face_recognizer)
            print "id: " + str(predicted_label) + ", name: " + names[predicted_label] + ", time: " + time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time())) + ", Confidence: " + str(predicted_confidence)
            if predicted_label < 0:
                cv2.putText(frame, 'Friend', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            else:
                cv2.putText(frame, names[predicted_label], (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time())), (x, y+h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


        cv2.imshow('camera', frame)
        if cv2.waitKey(10)&0xff == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
