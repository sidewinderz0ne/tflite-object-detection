# TensorFlow Lite Object Detection Training and Conversion

## Table of Contents
1. Installing Nvidia Driver, Cuda and cuDNN
2. Installing Anaconda and TensorFlow GPU
3. Preparing our Workspace and Anaconda Virtual Environment Directory Structure
4. Gathering and Labeling our Dataset
5. Generating Training Data
6. Configuring the Training Pipeline
7. Training the Model
8. Exporting the Inference Graph

## 1. Installing Nvidia Driver, Cuda and cudNN
a. Download and install compatible Nvidia driver (600mb) with cuda driver (2.7gb)
b. Download cuDNN 8 for cuda 10 (800mb) and cuDNN 8 for cuda 11 (800mb)
c. Extract and copy cuDNNs inside cuda

## 2. Installing Anaconda and TensorFlow GPU
a. Install Anaconda
b. Open Anaconda terminal
c. create a virtual environment with following commands

```
conda create -n tensorflow pip python=3.8
```

Then activate the environment with

```
conda activate tensorflow
```
**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

### Preparing our Workspace and Anaconda Virtual Environment Directory Structure
For the TensorFlow Object Detection API, there is a certain directory structure that we must follow to train our model. To make the process a bit easier, I added most of the necessary files in this repository.

Firstly, create a folder directly in C: and name it "TensorFlow". It's up to you where you want to put the folder, but you will have to keep in mind this directory path will be needed later to align with the commands. Once you have created this folder, go back to the Anaconda Prompt and switch to the folder with

```
cd C:\TensorFlow
```
Once you are here, you will have to clone the [TensorFlow models repository](https://github.com/tensorflow/models) with

```
git clone https://github.com/tensorflow/models.git
```
This should clone all the files in a directory called models. After you've done so, stay inside C:\TensorFlow and download [this](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector/archive/master.zip) repository into a .zip file. Then extract the two files, workspace and scripts, highlighted below directly in to the TensorFlow directory.
<p align="left">
  <img src="doc/clone.png">
</p>

Then, your directory structure should look something like this

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
└─ scripts/
└─ workspace/
   ├─ training_demo/
```
After we have setup the directory structure, we must install the prequisites for the Object Detection API. First we need to install the protobuf compiler with

```
conda install -c anaconda protobuf
```
Then you should cd in to the TensorFlow\models\research directory with

```
cd models\research
```
Then compile the protos with

```
protoc object_detection\protos\*.proto --python_out=.
```
After you have done this, close the terminal and open a new Anaconda prompt. If you are using the virtual environment we created earlier, then use the following command to activate it

```
conda activate tensorflow
```
With TensorFlow 2, pycocotools is a dependency for the Object Detection API. To install it with Windows Support use

```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
**Note that Visual C++ 2015 build tools must be installed and on your path, according to the installation instructions. If you do not have this package, then download it [here](https://go.microsoft.com/fwlink/?LinkId=691126).**

Go back to the models\research directory with 

```
cd C:\TensorFlow\models\research
```

Once here, copy and run the setup script with 

```
copy object_detection\packages\tf2\setup.py .
python -m pip install .
```
If there are any errors, report an issue, but they are most likely pycocotools issues meaning your installation was incorrect. But if everything went according to plan you can test your installation with

```
python object_detection\builders\model_builder_tf2_test.py
```
You should get a similar output to this

```
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 45.304s

OK (skipped=1)
```
This means we successfully set up the Anaconda Directory Structure and TensorFlow Object Detection API. We can now finally collect and label our dataset. So, let's go on to the next step!

### Gathering and Labeling our Dataset
Since the TensorFlow Object Detection API ready to go, we must collect and label pictures that the model will be trained and tested on. All the files that will be needed from
now on will be loacated in the workspace\training_demo directory. So take a second, look around, and get used to the structure of the directory. 

- ```annotations```: This is where we will store all our training data needed for our model. By this I mean the CSV and RECORD files needed for the training pipeline. There is also a PBTXT File with the labels for our model. If you are training your own dataset you can delete train.record and test.record, but if you are training my Pill Classifier model you can keep them.
- ```exported-models```: This is our output folder where we will export and store our finished inference graph.
- ```images```: This folder consists of a test and train folder. Here we will store the labelled images needed for training and testing as you can probably infer. The labelled images consist of the original image and an XML File. If you want to train the Pill Classifier model, you can keep the images and XML documents, otherwise delete the images and XML files.
- ```models```: In this folder we will store our training pipeline and checkpoint information from the training job as well as the CONFIG file needed for training.
- ```pre-trained-models```: Here we will store our pre-trained model that we will use as a starting checkpoint for training
- The rest of the scripts are just used for training and exporting the model, as well as a sample object detection scipt that performs inference on a test image.

If you want to train a model on your own custom dataset, you must first gather images. Ideally you would want to use 100 images for each class. Say for example, you are training a cat and dog detector. You would have to gather 100 images of cats and 100 images of dogs. For images of pills, I just looked on the internet and downloaded various images. But for your own dataset, I recommend taking diverse pictures with different backgrounds and angles.
<p align="left">
  <img src="doc/1c84d1d5-2318-5f9b-e054-00144ff88e88.jpg">
</p>
<p align="left">
  <img src="doc/5mg-325mg_Hydrocodone-APAP_Tablet.jpg">
</p>
<p align="left">
  <img src="doc/648_pd1738885_1.jpg">
</p>

After gathering some images, you must partition the dataset. By this I mean you must seperate the data in to a training set and testing set. You should put 80% of your images in to the images\training folder and put the remaining 20% in the images\test folder. After seperating your images, you can label them with [LabelImg](https://tzutalin.github.io/labelImg).


After Downloading LablelImg, configure settings such as the Open Dir and Save Dir. This let's you cycle through all the images and create bounding boxes and labels around the objects. Once you have labelled your image make sure to save and go on to the next image. Do
this for all the images in the images\test and images\train folders. 

<p align="left">
  <img src="doc/labelimg.png">
</p>

We have now gathered our dataset. This means we are ready to generate training data. So onwards to the next step!

### Generating Training Data

Since our images and XML files are prepared, we are ready to create the label_map. It is located in the annotations folder, so navigate to that within File Explorer. After you've located label_map.pbtxt, open it with a Text Editor of your choice. If you plan to use my Pill Classification Model, you don't need to make any changes and you can skip to configuring the pipeline. If you want to make your own custom object detector you must create a similar item for each of your labels. Since my model had two classes of pills, my labelmap looked like 
```
item {
    id: 1
    name: 'Acetaminophen 325 MG Oral Tablet'
}

item {
    id: 2
    name: 'Ibuprofen 200 MG Oral Tablet'
}
```
For example, if you wanted to make a basketball, football, and baseball detector, your labelmap would look something like
```
item {
    id: 1
    name: 'basketball'
}

item {
    id: 2
    name: 'football'
}

item {
    id: 3
    name: 'baseball'
}
```
Once you are done with this save as ```label_map.pbtxt``` and exit the text editor. Now we have to generate RECORD files for training. The script to do so is located in C:\TensorFlow\scripts\preprocessing, but we must first install the pandas package with

```
pip install pandas
```
Now we should navigate to the scripts\preprocessing directory with

```
cd C:\TensorFlow\scripts\preprocessing
```

Once you are in the correct directory, run these two commands to generate the records

```
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\train -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\train.record

python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\test -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\test.record
```
 After each command you should get a success message stating that the TFRecord File has been created. So now under ```annotations``` there should be a ```test.record``` and ```train.record```. That means we have generated all the data necessary, and we can proceed to configure the training pipeline in the next step

### Configuring the Training Pipeline
For this tutorial, we will use a CONFIG File from one of the TensorFlow pre-trained models. There are plenty of models in the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), but we will use the [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz), as it is on the faster end of the spectrum with decent performance. If you want you can choose a different model, but you will have to alter the steps slightly.

To download the model you want, just click on the name in the TensorFlow Model Zoo. This should download a tar.gz file. Once it has downloaded, extracts the contents of the file to the ```pre-trained-models``` directory. The structure of that directory should now look something like this

```
training_demo/
├─ ...
├─ pre-trained-models/
│  └─ ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```
Now, we must create a directory to store our training pipeline. Navigate to the ```models``` directory and create a folder called ```my_ssd_mobilenet_v2_fpnlite```. Then copy the ```pipeline.config``` from the pre-trained-model we downloaded earlier to our newly created directory. Your directory should now look something like this

```
training_demo/
├─ ...
├─ models/
│  └─ my_ssd_mobilenet_v2_fpnlite/
│     └─ pipeline.config
└─ ...
```

Then open up ```models\my_ssd_mobilenet_v2_fpnlite\pipeline.config``` in a text editor because we need to make some changes.
- Line 3. Change ```num_classes``` to the number of classes your model detects. For the basketball, baseball, and football, example you would change it to ```num_classes: 3```
- Line 135. Change ```batch_size``` according to available memory (Higher values require more memory and vice-versa). I changed it to:
  - ```batch_size: 6```
- Line 165. Change ```fine_tune_checkpoint``` to:
  - ```fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"```
- Line 171. Change ```fine_tune_checkpoint_type``` to:
  - ```fine_tune_checkpoint_type: "detection"```
- Line 175. Change ```label_map_path``` to:
  - ```label_map_path: "annotations/label_map.pbtxt"```
- Line 177. Change ```input_path``` to:
  - ```input_path: "annotations/train.record"```
- Line 185. Change ```label_map_path``` to:
  - ```label_map_path: "annotations/label_map.pbtxt"```
- Line 189. Change ```input_path``` to:
  - ```input_path: "annotations/test.record"```

Once we have made all the necessary changes, that means we are ready for training. So let's move on to the next step!
### Training the Model
Now you go back to your Anaconda Prompt. ```cd``` in to the ```training_demo``` with 

```
cd C:\TensorFlow\workspace\training_demo
```

I have already moved the training script in to the directory, so to run it just use 

```
python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config
```

When running the script, you should expect a few warnings but as long as they're not errors you can ignore them. Eventually when the training process starts you should see output similar to this

```
INFO:tensorflow:Step 100 per-step time 0.640s loss=0.454
I0810 11:56:12.520163 11172 model_lib_v2.py:644] Step 100 per-step time 0.640s loss=0.454
```

Congratulations! You have officially started training your model! Now you can kick back and relax as this will take a few hours depending on your system. With my specs that I mentioned earlier, training took about 2 hours. TensorFlow logs output similar to the one above every 100 steps of the process so if it looks frozen, don't worry about it. This output shows you two statistics: per-step time and loss. You're going to want to pay attention to the loss. In between logs, the loss tends to decrease. Your ideally going to want to stop the program when it's between 0.150 and 0.200. This prevents underfitting and overfitting. For me it took around 4000 steps before the loss entered that range. And then to stop the program just use CTRL+C.

### Monitoring Training with TensorBoard (Optional)

TensorFlow allows you to monitor training and visualize training metrics with TensorBoard! Keep in mind this is completely optional and wont affect the training process, so it's up to you whether you want to do it. 
First, open up a new Anaconda Prompt. Then activate the virtual environment we configured with

```
conda activate tensorflow
```

Then ```cd``` in to the ```training_demo``` directory with

```
cd C:\TensorFlow\workspace\training_demo
```
To start a TensorBoard Server, use 
 
```
tensorboard --logdir=models\my_ssd_mobilenet_v2_fpnlite
```
It should output something like this
 
```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then just open up a web browser and paste the URL given in to the search bar. This should take you to the TensorBoard Server where you can continuously monitor training!

### Exporting the Inference Graph

Once you have finished training and stopped the script, you are ready to export your finished model! You should still be in the ```training_demo``` directory but if not use

```
cd C:\TensorFlow\workspace\training_demo
```

I have already moved the script needed to export, so all you need to do is run this command

```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model
```

**Note that if you get an error similar to ```TypeError: Expected Operation, Variable, or Tensor, got block4 in exporter_main_v2.py``` look at [this](https://github.com/tensorflow/models/issues/8881) error topic**

But if this program finishes successfully, then congratulations because your model is finished! It should be located in the ```C:\TensorFlow\workspace\training_demo\exported-models\my_mobilenet_model\saved_model``` folder. There should be an PB File called ```saved_model.pb```. This is the inference graph! I also prefer to copy the ```label_map.pbtxt``` file in to this directory because it makes things a bit easier for testing. If you forgot where the labelmap is located it should be in ```C:\TensorFlow\workspace\training_demo\annotations\label_map.pbtxt```. Since the labelmap and inference graph are organized, we are ready to test! 

### Evaluating the Model (Optional)

If you want to measure model metrics such as IoU, mAP, Recall, and Precision, you'll want to complete this step. The most up-to-date TensorFlow Documentation for evaluating the model will be located [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md#evaluation)

You should still be in your ```TensorFlow/workspace/training_demo``` but if not cd into it with

```
cd C:\TensorFlow\workspace\training_demo
```

Now just run the following command to following command for evaluation

```
python model_main_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --model_dir models\my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```

**Note that if you get an error similar to ```TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer```, just downgrade your NumPy version. For me, version 1.17.3 worked so you can install it with ```pip install numpy==1.17.3```**

If everything works properly, you should get something similar to this

<p align="center">
  <img src="doc/evaluation.png">
</p>

### Testing out the Finished Model

To test out your model, you can use the sample object detection script I provided called ```TF-image-od.py```. This should be located in ```C:\TensorFlow\workspace\training_demo```. **Update**: I have added video support, argument support, and an extra OpenCV method. The description for each program shall be listed below 
- ```TF-image-od.py```: This program uses the viz_utils module to visualize labels and bounding boxes. It performs object detection on a single image, and displays it with a cv2 window.
- ```TF-image-object-counting.py```: This program also performs inference on a single image. I have added my own labelling method with OpenCV which I prefer. It also counts the number of detections and displays it in the top left corner. The final image is, again, displayed with a cv2 window.
- ```TF-video-od.py```: This program is similar to the ```TF-image-od.py```. However, it performs inference on each individual frame of a video and displays it via cv2 window.
- ```TF-video-object-counting.py```: This program is similar to ```TF-image-object-counting.py``` and has a similar labelling method with OpenCV. Takes a video for input, and also performs object detection on each frame, displaying the detection count in the top left corner.

The usage of each program looks like 

```
usage: TF-image-od.py [-h] [--model MODEL] [--labels LABELS] [--image IMAGE] [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Folder that the Saved Model is Located In
  --labels LABELS       Where the Labelmap is Located
  --image IMAGE         Name of the single image to perform detection on
  --threshold THRESHOLD Minimum confidence threshold for displaying detected objects
```
If the model or labelmap is located anywhere other than where I put them, you can specify the location with those arguments. You must also provide an image/video to perform inference on. If you are using my Pill Detection Model, this is unecessary as the default value should be fine. If you are using one of the video scripts, use ```--video``` instead of ```--image``` and provide the path to your test video. For example, the following steps run the sample ```TF-image-od.py``` script.

```
cd C:\TensorFlow\workspace\training_demo
```

Then to run the script, just use

```
python TF-image-od.py
``` 

**Note that if you get an error similar to ```
cv2.error: OpenCV(4.3.0) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-kv3taq41\opencv\modules\highgui\src\window.cpp:651: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
``` just run ```pip install opencv-python``` and run the program again**

If everything works properly you should get an output similar to this
<p align="center">
  <img src="doc/output.png">
</p>

This means we're done! Over the next few weeks or months, I'll keep working on new programs and keep testing! If you find something cool, feel free to share it, as others can also learn! And if you have any errors, just raise an issue and I'll be happy to take a look at it. Congratulations, and until next time, bye!

# Converting TensorFlow Models to TensorFlow Lite
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
### This Guide Contains Everything you Need to Convert Custom and Pre-trained TensorFlow Models to TensorFlow Lite
Following these intstructions, you can convert either a custom model or convert a pre-trained TensorFlow model. If you want to train a custom TensorFlow object detection model, I've made a detailed [GitHub guide](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector) and a YouTube video on the topic.

[![Link to my vid](https://github.com/armaanpriyadarshan/Object-Detection-on-Raspberry-Pi/blob/master/doc/Thumbnail2.png)](https://www.youtube.com/watch?v=oqd54apcgGE)

**The following steps for conversion are based off of the directory structure and procedures in this guide. So if you haven't already taken a look at it, I recommend you do so.
To move on, you should have already**
  - **Installed Anaconda**
  - **Setup the Directory Structure**
  - **Compiled Protos and Setup the TensorFlow Object Detection API**
  - **Gathered Training Data**
  - **Trained your Model (without exporting)**
  
## Steps
1. [Exporting the Model](https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-Conversion.md#exporting-the-model) 
2. [Creating a New Environment and Installing TensorFlow Nightly](https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-Conversion.md#creating-a-new-environment-and-installing-tensorflow-nightly)
3. [Converting the Model to TensorFlow Lite](https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-Conversion.md#converting-the-model-to-tensorflow-lite)
4. [Preparing our Model for Use](https://github.com/armaanpriyadarshan/TensorFlow-2-Lite-Object-Detection-on-the-Raspberry-Pi/blob/main/TFLite-Conversion.md#preparing-our-model-for-use)
 
### Exporting the Model
Assuming you followed my previous guide, your directory structure should look something like this
<p align="left">
  <img src="doc/Screenshot 2020-11-16 104855.png">
</p>

If you haven't already, make sure you have already configured the training pipeline and trained the model. You should now have a training directory (if you followed my other guide, this is ```models\my_ssd_mobilenet_v2_fpnlite```) and a ```pipeline.config``` file (```models\my_ssd_mobilenet_v2_fpnlite\pipeline.config```). Open up a new Anaconda terminal and activate the virtual environment we made in the other tutorial with

```
conda activate tensorflow
```
Now, we can change directories with

```
cd C:\TensorFlow\workspace\training_demo
```
Now, unlike my other guide, we aren't using ```exporter_main_v2.py``` to export the model. For TensorFlow Lite Models, we have to use ```export_tflite_graph_tf2.py```. You can export the model with
```
python export_tflite_graph_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --output_directory exported-models\my_tflite_model
```
**Note: At the moment, TensorFlow Lite only support models with the SSD Architecture (excluding EfficientDet). Make sure that you have trained with an SSD training pipeline before you continue. You can take a look at the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) or the [documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md) for the most up-to-date information.**

### Creating a New Environment and Installing TensorFlow Nightly
To avoid version conflicts, we'll first create a new Anaconda virtual environment to hold all the packages necessary for conversion. First, we must deactivate our current environment with

```
conda deactivate
```

Now issue this command to create a new environment for TFLite conversion.

```
conda create -n tflite pip python=3.7
```

We can now activate our environment with

```
conda activate tflite
```

**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

Now we must install TensorFlow in this virtual environment. However, in this environment we will not just be installing standard TensorFlow. We are going to install tf-nightly. This package is a nightly updated build of TensorFlow. This means it contains the very latest features that TensorFlow has to offer. There is a CPU and GPU version, but if you are only using it conversion I'd stick to the CPU version because it doesn't really matter. We can install it by issuing

```
pip install tf-nightly
```
Now, to test our installation let's use a Python terminal.
```
python
```
Then import the module with
```
Python 3.7.9 (default, Aug 31 2020, 17:10:11) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version)
```

**Note: You might get an error with importing the newest version of Numpy. It looks something like this ```RuntimeError: The current Numpy installation ('D:\\Apps\\anaconda3\\envs\\tflite\\lib\\site-packages\\numpy\\__init__.py') fails to pass a sanity check due to a bug in the windows runtime. See this issue for more information: https://tinyurl.com/y3dm3h86```. You can fix this error by installing a previous version of Numpy with ```pip install numpy==1.19.3```.**

If the installation was successful, you should get the version of tf-nightly that you installed. 
```
2.5.0-dev20201111
```

### Converting the Model to TensorFlow Lite
Now, you might have a question or two. If the program is called ```export_tflite_graph_tf2.py```, why is the exported inference graph a ```saved_model.pb``` file? Isn't this the same as standard TensorFlow?
<p align="left">
  <img src="doc/saved_model.png">
</p>

Well, in this step we'll be converting the ```saved_model``` to a single ```model.tflite``` file for object detection with tf-nightly. I recently added a sample converter program to my other repository called ```convert-to-tflite.py```. This script takes a saved_model folder for input and then converts the model to the .tflite format. Additionally, it also quantizes the model. If you take a look at the code, there are also various different features and options commented. These are optional and might be a little buggy. For some more information, take a look at the [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert/). The usage of this program is as so

```
usage: convert-to-tflite.py [-h] [--model MODEL] [--output OUTPUT]

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    Folder that the saved model is located in
  --output OUTPUT  Folder that the tflite model will be written to
```

At the moment I'd recommend not using the output argument and sticking to the default values as it still has a few errors. Enough talking, to convert the model run
```
python convert-to-tflite.py
```

You should now see a file in the ```exported-models\my_tflite_model\saved_model``` directory called ```model.tflite```

<p align="left">
  <img src="doc/model.tflite.png">
</p>

Now, there is something very important to note with this file. Take a look at the file size of the ```model.tflite``` file. **If your file size is 1 KB, that means something has gone wrong with conversion**. If you were to run object detection with this model, you will get various errors. As you can see in the image, my model is 3,549 KB which is an appropriate size. If your file is significantly bigger, 121,000 KB for example, it will drastically impact performance while running. With a model that big, my framerates dropped all the way down to 0.07 FPS. If you have any questions about this, feel free to raise an issue and I will try my best to help you out. 

### Preparing our Model for Use
Now that we have our model, it's time to create a new labelmap. Unlike standard TensorFlow, TensorFlow uses a .txt file instead of a .pbtxt file. Creating a new labelmap is actually much easier than it sounds. Let's take a look at an example. Below, I have provided the ```label_map.pbtxt``` that I used for my Pill Detection model.
```
item {
    id: 1
    name: 'Acetaminophen 325 MG Oral Tablet'
}

item {
    id: 2
    name: 'Ibuprofen 200 MG Oral Tablet'
}
```
If we were to create a new labelmap for TensorFlow Lite, all we have to do is write each of the item names on it's own line like so
```
Acetaminophen 325 MG Oral Tablet
Ibuprofen 200 MG Oral Tablet
```
Once you are finished filling it out save the file within the ```exported-models\my_tflite_model\saved_model``` as ```labels.txt```. The directory should now look like this

<p align="left">
  <img src="doc/final model.png">
</p>

We're done! The model is now ready to be used. If you want to run on the Raspberry Pi, you can transfer the model any way you prefer. [WinSCP](https://winscp.net/eng/index.php), an SFTP client, is my favorite method. Place the ```model.tflite``` file and the ```labels.txt``` in the ```tensorflow/models``` directory on the Raspberry Pi. Once your done, it should look like this

<p align="left">
  <img src="doc/folder.png">
</p>

There you go, you're all set to run object detection on the Pi! Good Luck!

#ADDING METADA FOR ANDROID
```
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

ObjectDetectorWriter = object_detector.MetadataWriter
_MODEL_PATH = "exported-models\my_tflite_model\model.tflite"
_LABEL_FILE = "write_metadata\labelmap.txt"
_SAVE_TO_PATH = "write_metadata\detect.tflite"

writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [127.5], [127.5], [_LABEL_FILE])
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

# Verify the populated metadata and associated files.
displayer = metadata.MetadataDisplayer.with_model_file(_SAVE_TO_PATH)
print("Metadata populated:")
print(displayer.get_metadata_json())
print("Associated file(s) populated:")
print(displayer.get_packed_associated_file_list())
```

