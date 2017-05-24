## fast-style-transfer webcam script

This is a fork of [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) which has an additional script, `run_webcam.py` to apply style models live to a webcam stream. Go to the README of the original page for instructions on how to train your own models, apply them to images and movies, and all the original functionality of that repository.

### Installation

 - [CUDA](https://developer.nvidia.com/cuda-downloads) + [CuDNN](https://developer.nvidia.com/cudnn)
 - [TensorFlow](https://www.tensorflow.org/install/) GPU-enabled
 - [OpenCV](https://pypi.python.org/pypi/opencv-python) (this is tested on cv 2.4, not most recent, but presumably both work)


### Setting up models

At the top of the file `run_webcam.py`, there are paths to model files and style images in the variable list `models`. They are not included in the repo because of space. If you'd like to use the pre-trained models referred to up there, these models may be [downloaded from this shared folder](https://drive.google.com/open?id=0B3WXSfqxKDkFUFl3YllzS1ZqbkU). To train your own, refer to the [original documentation](https://github.com/lengstrom/fast-style-transfer).

### Usage

    python run_webcam.py --width 360 --disp_width 800 --display_source 1

There are three arguments:

 - `width` refers to the width in pixels of the image being restyled (the webcam will be scaled down or up to this size).  
 - `disp_width` is the width in pixels of the image to be shown on the screen. The restyled image is resized to this after being generated. Having `disp_width` > `width` lets you run the model more quickly but generate a bigger image of lesser quality.
 - `display_source` is whether or not to display the content image (webcam) and corresponding style image alongside the output image (1 by default, i.e. True)

### Example

![stylized webcam](styles/stylenet_webcam.gif)