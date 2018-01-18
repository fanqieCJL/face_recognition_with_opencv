import cv2
import os
import numpy as np
from argparse import ArgumentParser

def read_images(data_dir, resize=False):
    images = []
    labels = []
    names = []
    for dir_path, dir_names, file_names in os.walk(data_dir):
        for i, sub_dir_name in enumerate(dir_names):
            names.append(sub_dir_name)
            subject_path = os.path.join(dir_path, sub_dir_name)
            for file_name in os.listdir(subject_path):
                if file_name.split('.')[-1] != 'pgm':
                    continue
                file_path = os.path.join(subject_path, file_name)
                print 'reading image: {0}'.format(file_path)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if resize:
                    img = cv2.resize(img, (360, 360))
                images.append(img)
                labels.append(i)
    # images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int32)
    return images, labels, names

def save_names(names, out_dir):
    namesfile_path = os.path.join(out_dir, 'names.txt')
    with open(namesfile_path, mode='w') as f:
        for name in names:
            f.write(name + '\n')

def LBPH(data_dir, out_dir):
    face_recognizer = cv2.createLBPHFaceRecognizer()
    print 'start reading images in {0}'.format(data_dir)
    images, labels, names = read_images(data_dir, resize=False)

    save_names(names, out_dir)

    print 'start training face recognizer'
    face_recognizer.train(images, labels)

    model_path = os.path.join(out_dir, 'LBPH.yml')# LBPH.yml, Eigen.yml, Fisher.yml
    print 'save trained model to {0}'.format(model_path)
    face_recognizer.save(model_path)

def Eigen(data_dir, out_dir):
    face_recognizer = cv2.createEigenFaceRecognizer(5)
    print 'start reading images in {0}'.format(data_dir)
    images, labels, names = read_images(data_dir, resize=True)

    save_names(names, out_dir)

    print 'start training face recognizer'
    face_recognizer.train(images, labels)

    model_path = os.path.join(out_dir, 'Eigen.yml')# LBPH.yml, Eigen.yml, Fisher.yml
    print 'save trained model to {0}'.format(model_path)
    face_recognizer.save(model_path)


def Fisher(data_dir, out_dir):
    face_recognizer = cv2.createFisherFaceRecognizer()
    print 'start reading images in {0}'.format(data_dir)
    images, labels, names = read_images(data_dir, resize=True)

    save_names(names, out_dir)

    print 'start training face recognizer'
    face_recognizer.train(images, labels)

    model_path = os.path.join(out_dir, 'Fisher.yml')# LBPH.yml, Eigen.yml, Fisher.yml
    print 'save trained model to {0}'.format(model_path)
    face_recognizer.save(model_path)

def main():
    parser = ArgumentParser()
    parser.add_argument('-dd', '--data-directory', dest='data_dir', type=str, default='./data',
                        help='location root of your data ')
    parser.add_argument('-o', '--out-directory', dest='out_dir', type=str, default='./models',
                        help='a directory where to save the trained model')
    parser.add_argument('-m', '--model', dest='model', type=str, default='LBPH',
                        help='face recognizer model name: LBPH, Eigen, Fisher, Both')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    model = args.model

    if model == 'LBPH':
        LBPH(data_dir, out_dir)
    elif model == 'Eigen':
        Eigen(data_dir, out_dir)
    elif model == 'Fisher':
        Fisher(data_dir, out_dir)
    elif model == 'Both':
        LBPH(data_dir, out_dir)
        Eigen(data_dir, out_dir)
        Fisher(data_dir, out_dir)
    else:
        raise Exception('model name error!')

if __name__ == '__main__':
    main()
