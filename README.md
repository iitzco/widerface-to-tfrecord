# Wider Face Dataset to TF Record

Convert the [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) to TFRecord format.

> **Important**: this project runs under Python3.

## Why doing this?

The TFRecord format will allow you to use the Tensorflow Object Detection API in order to train a new model based on Transfer Learning from many available pretrained models.

## How to run

1. Clone the repo
2. [Optional] Create a virtual environment to keep dependencies isolated.
3. Run `pip install -r requirements.txt`
4. Configure `config.py` file with your appropiate paths.
   * You should have downloaded from [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) the dataset and the annotations' folder
   * If you don't want a set to be converted then leave the field in `None`. For example, if *test* is not going to be converted you should leave `TEST_WIDER_PATH` in `None`.
5. Run `python wider_to_tfrecord.py`
6. When done, `tfrecord` files should be in the output folder you specified in `config.py`.


## About testing images set

Testing images do not contain ground truth bounding boxes. In case you want to convert the testing set as well, the `test.tfrecord` file will contain images only without bounding boxes.

## Acknowledgement

* This work was based on [this](https://github.com/yeephycho/widerface-to-tfrecord) original version for Python2.
* The conversion follows the standards indicated in the official Tensorflow Object Detection API site [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
