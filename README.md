## Project Description:
Steering prediction (behavioral cloning) project on Udacity simulator. We trained a CNN model to predict steering angles of frames pixels from a single on-board camera. Udacity had been released a self driving car simulator that could be used for collecting a dataset that consists of frames of on-board camera and their steering angles, speed and throttle. Using this collected dataset, we trained a CNN model by mapping frames pix- els from a single on-board camera to predict their steering angles. This CNN model architecture consists of three convolutional layers with three fully connected layers that inspired from NVIDIA’s CNN architecture. Using small CNN network architecture minimized the processing time and computational cost.






## Model Architecture
This CNN model consists of three convolutional layers and two fully connected layers the input layer is a raw RGB image and the output layer is the prediction steering angle of the input image. The first convolutional layer uses a 9×9 kernel and a 4×4 stride and the other two convolutional layers use a 5×5 kernel and a 2×2 stride, which is mainly for steering angle prediction.
<p align="center">
 <img  src="https://github.com/anasbadawy/Steering-Prediction-CNN/blob/master/model%20diagram/last.jpeg">
</p>


##  Dataset Creation

- For creating a convolutional neural network models that is able to predict steering angles of autonomous vehicles using Udacity simulator, a dataset of frames that have variety of roads and lighting is extracted along with the steering value of each frame. 

- These images is obtained by Udacity simulator and they are 21000 of images and labeled by their steering values. 

- They were approximately 1 hour worth of driving data by driving the car in the center of the road. The steering angles are in degrees. So it has been normalized in the [-1,1] range. Because of the dataset was unbalanced because most of it has degrees between [-5,5] which is the format of Udacity simulator. 

- We applied a sampling process to balance this dataset by splitting steering angles into 1000 bins and using at most 300 frames for each bin. Some of bins which are more than 0.4 after normalization couldn’t react 300 frames because they were so rare. After this sampling process. the dataset became 8300 samples. 

- We split them into 80 percent for training and 20 percent for testing sets.

- For increasing data, we applied data augmentation technique by creating more images from existed ones by randomly adding vertical shadow over images to be similar to shadows on real roads.


##  Experiments

- For training CNN model, we used Adam optimizer with learning rate of 1e-04 as an opti- mization function and and MSE (mean square error) as a loss function. Moreover, we preprocess our data by cropping top and bottom of each frame, resizing it to be 32x128 which is the shape that our model expects and also scaling pixels values to [0,1].

- After preprocessing data, we trained our model using the 21000 samples dataset without using balancing data technique by applying 1000 bins and a maximum of 300 frames for each bin and also we trained it using balancing data technique which has been mentioned before. We trained both of them 20 epochs. The MSE of them has been as shown below.

- Experiment emphasizes that balancing data technique was better for cleaning data and let CNN model performance to be more accurate and efficient for predicting steering angles of the given inputs.

<p align="center">
 <img  src="https://github.com/anasbadawy/Steering-Prediction-CNN/blob/master/model%20diagram/table.png">
</p>
