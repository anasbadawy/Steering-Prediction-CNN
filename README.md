# Steering Prediction CNN

## Project Description:
Steering prediction (behavioral cloning) project on Udacity simulator. We trained a CNN model to predict steering angles of frames pixels from a single on-board camera. Udacity had been released a self driving car simulator that could be used for collecting a dataset that consists of frames of on-board camera and their steering angles, speed and throttle. Using this collected dataset, we trained a CNN model by mapping frames pix- els from a single on-board camera to predict their steering angles. This CNN model architecture consists of three convolutional layers with three fully connected layers that inspired from NVIDIA’s CNN architecture. Using small CNN network architecture minimized the processing time and computational cost.


## Model Architecture
This CNN model consists of three convolutional layers and two fully connected layers the input layer is a raw RGB image and the output layer is the prediction steering angle of the input image. The first convolutional layer uses a 9×9 kernel and a 4×4 stride and the other two convolutional layers use a 5×5 kernel and a 2×2 stride, which is mainly for steering angle prediction.

<p align="center">
 <img src="https://github.com/anasbadawy/Steering-Prediction-CNN/blob/master/model%20diagram/last.png">
</p>
