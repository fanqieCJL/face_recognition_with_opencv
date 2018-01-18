Face detection and recognition using OpenCV

## 概述 (Overview)
该项目基于OpenCV中的人脸检测(`cv2.CascadeClassifier`)、识别(`cv2.createLBPHFaceRecognizer`)方法搭建了一个人脸识别系统。该系统功能包括：
* 人脸图像录入到数据库（图像文件）
* 训练人脸识别模型（三种模型，Eigen、Fisher、LBPH）
* 加载训练后的模型进行实时人脸识别  

### 基本使用流程为：
1. 多次调用`record_one_face.py`脚本录入多个人的脸部图像

  ```python record_one_face.py --name=person_name```

2. 运行`trainer.py`训练人脸识别模型

  ```python trainer.py --model=LBPH```

3. 加载训练后的模型实时人脸识别

  ```python main.py --model=LBPH```

### 运行条件：

* Python 2.x
* OpenCV 2.x
* numpy
* Camera

## 项目文件分布

### main.py

这是演示脚本入口，该脚本实时检测人脸并进行识别。

### record_one_face.py

该脚本用来添加一个新的人脸，执行：

```
python record_one_face.py --name=gaoyan
```

该脚本自动调用人脸检测模型，并将检测到的人脸以`name.pgm`图片格式保存在`./data/name`路径下。
以上面执行脚本命令为例，你会得到一些（默认100张）人脸图片保存在`./data/mingzi/`目录下。该目录作为姓名为mingzi的人脸图像数据

### trainer.py

该脚本在准备好所有人脸数据之后用来训练人脸识别的模型，模型包括三种基本方法，分别为Eigen、Fisher和LBPH，默认为LBPH方法。这种方法不需要resize人脸图像，其余两种方法在用作训练之前必须resize到统一尺寸`(200, 2000)`。

训练好的模型会被自动保存到`./models/`路径下，三种模型对应的文件名分别为

* `LBPH.yml`
* `Eigen.yml`
* `Fisher.yml`

同时，该脚本会保存`names.txt`文件，位置和模型保存位置一致，在`./models/`路径下。该文件是当前所有脸部图像的姓名，顺序与`./data/`路径下各子目录名一致（顺序一致，姓名一致）。
这些训练好的模型在运行`main.py`演示实时人脸检测时会被加载。

### cascades

该目录下保存了OpenCV中用于人脸检测算法的模型参数，`haarcascade_frontalface_default.xml`在检测人脸的时候会被`cv2.CascadeClassifier()`方法调用，用以创建一个face detector。这些xml文件来自OpenCV项目下的`data./haarcascades/`。你可以在Github上下载OpenCV项目源码。

### data

如前面所述，该目录下保存了所有`record_one_face.py`脚本执行后检测到的人脸图像。每个子目录文件夹名为该文件夹下保存的人脸对应的人物名。形式如下：

./data
|------name1/
---------------0.pgm
---------------1.pgm
---------------100.pgm

|------name2/

|------name99/
...

### models

如前所述，该目录下保存了所有`trainer.py`训练后的人脸识别模型文件。
