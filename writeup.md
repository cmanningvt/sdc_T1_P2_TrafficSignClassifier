# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/loss_plot_initial.png "loss_plot_initial"
[image2]: ./report_images/loss_plot_final.png "loss_plot_final"
[image3]: ./report_images/sign_types.png "sign_types"
[image4]: ./report_images/bar_chart.png "bar_chart"
[image5]: ./custom_images/sign_01.jpg "Traffic Sign 1"
[image6]: ./custom_images/sign_02.jpg "Traffic Sign 2"
[image7]: ./custom_images/sign_03.jpg "Traffic Sign 3"
[image8]: ./custom_images/sign_04.jpg "Traffic Sign 4"
[image9]: ./custom_images/sign_05.jpg "Traffic Sign 5"
[image10]: ./custom_images/sign_06.jpg "Traffic Sign 6"
[image11]: ./custom_images/sign_07.jpg "Traffic Sign 7"
[image12]: ./custom_images/sign_08.jpg "Traffic Sign 8"
[image13]: ./custom_images/sign_09.jpg "Traffic Sign 9"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted

#### 1. The project submission includes all required files.

All files are included included. Here is a link to my [project](https://github.com/cmanningvt/sdc_T1_P2_TrafficSignClassifier)

### Data Set Summary & Exploration

#### 1. The submission includes a basic summary of the data set.

I began by importing the 'signnames.csv' file and parsing and displaying the data. This was very useful for me because it allowed me to view signs based on name rather than a numeric classId.

Next, I used the numpy library to gether the sizes of the training, validation, and testing data sets.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. The submission includes an exploratory visualization on the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the different data sets and output labels.

![alt text][image4]

The different sign types seem very well distributed amongst the three different data sets.

I also included a figure showing the first type of each sign to appear in the training data set. Note that some of these are very dark and not easily recognizable. Improving this would have to be done after the network is fully trained and operational

![alt text][image3]

### Design and Test a Model Architecture

#### 1. The submission describes the preprocessing techniques used and why these techniques were chosen.

I decided not to convert the images to grayscale because traffic signs have standardized coloring. It is important to remove color dependencies when we want our network to be unbiased based on color. However, since color is an important part of the traffic sign definition, I decided to leave the images in 3 color channels. I did experiment with grayscaling the images when I was having issues with my network in the beginning.

As a last step, I normalized the image data because it helps the CNN training process when data is centralized and normalized.

I did not choose to augment the data set because I did not run into scenarios where I needed additional data. In the future, this could help with different lighting, weather, and aging conditions though. Orientation augmenting does not seem to make a lot of sense to me because the traffic signs are almost always in an upright position.

#### 2. The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten				| Outputs 400x1									|
| Fully connected		| Outputs 120x1									|
| RELU					|												|
| Dropout				| Keep_prob = 0.8 during training				|
| Fully connected		| Outputs 84x1									|
| RELU					|												|
| Dropout				| Keep_prob = 0.8 during training				|
| Fully connected		| Outputs n_classesx1							|
| Softmax				| etc.        									|
 
This was started by using the LeNet architecture from the previous lab to begin with. After initial training however, I was running into issues with overfitting. My validation accuracy was significantly lower than my testing accuracy. 

![alt text][image1]

There are two changes I made to address this. The first was to increase my epochs significantly. I had initially been underestimating the stability of my training loss outputs after each epoch. Extending the epochs to 100 showed improvement on the training loss calculation to near 0. The second improvement was to add the two dropout layers inbetween the fully connected layers in my CNN. This helped to avoid overfitting as well.

#### 3. The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

All optimizers and hyperparameters used were based on the LeNet lab from the previous lesson.

The optimizer used was the AdamOptimizer.

Mu and sigma were used in the initialization of the weights for the CNN layers. All biases were iniatialized to zero.

| Parameters         	|     Values	        						| 
|:---------------------:|:---------------------------------------------:| 
| Epochs         		| 100   										| 
| Batch size     		| 128											|
| Learning Rate			| 0.0005										|
| Mu	      			| 0	 											|
| Sigma					| 0.2											|
| Keep_prob				| 0.8											|

#### 4. The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

My final model results were:
* training set accuracy of 0.99974
* validation set accuracy of 0.94535 
* test set accuracy of 0.935

![alt text][image2]

As mentioned previously, the LeNet architecture was chosen for this. This had already proven to be an easy to use architecture for recognizing the image data from the previous lesson. The final data proves that this network is working well.

### Test a Model on New Images

#### 1. The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

I originially chose to test on 7 images shown here:


![alt text][image5] 

![alt text][image6] 

![alt text][image7] 

![alt text][image8]

![alt text][image9] 

![alt text][image10]

![alt text][image11]


However, after initial success with these images I tried to add some more difficult images. This one is an image from night time with a light directly above, so the lighting in this image is not standard.


![alt text][image12] 


This one is from night time as well and partially covered in snow. I was curious to see how my network would react to conditions like these.


![alt text][image13]


#### 2. The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

Here are the results of the prediction:

| Image			        					|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection		| Right-of-way at the next intersection   		| 
| Road work     							| Road work 									|
| Keep right								| Keep right									|
| Speed limit (30km/h)	      				| Speed limit (30km/h)					 		|
| Priority road								| Priority road      							|
| Turn right ahead     						| Turn right ahead 								|
| No entry									| No entry										|
| Traffic signals	      					| **General caution						 		|
| Road work									| Road work      								|


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.889%. This compares favorably to the accuracy on the test set of 93.5%

#### 3. The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000000000000e+00	| Right-of-way at the next intersection			| 
| 7.078847750621819e-23	| Beware of ice/snow							|
| 1.438273543186304e-26	| Double curve									|
| 4.986559413448082e-31	| General caution								|
| 6.982266584950694e-32 | Pedestrians									|

For the second image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.993367791175842e-01	| Road work										| 
| 6.630903226323426e-04	| Bicycles crossing								|
| 1.536754581366040e-07	| Wild animals crossing							|
| 2.654338138174950e-10	| Bumpy road									|
| 5.348664219861909e-13 | Beware of ice/snow							|

This is the first image that shows even a hint of doubt. But these other possibilities are still very very small compared to the highest output.

Images 3-7 show similar trends to the first two and can be viewed in the python notebook or html file.


For the eigth image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.641582727432251e-01	| General caution								| 
| 1.358417272567749e-01	| Traffic signals								|
| 1.352573830715187e-09	| Bicycles crossing								|
| 7.784226388474202e-12	| Road narrows on the right						|
| 2.031765013326445e-15 | Pedestrians									|

This is the only image to be classified incorrectly. And you can notice that the correct classification has the second highest probability of ~14%. Perhaps some data augmentation to account for different lighting conditions would have been useful in correctly clasifying the image.

For the 9th image, there are two dominant choices. The first being the correct classification with a probability of ~77 percent. The second option having a probability of ~23 percent. All other options are significantly lower.

Again, these last two images were interesting tests for me and proved to be some good insight into my network. Instead of just testing standard or easy images, I tried for images that were a little harder.




Overall, I feel that my CNN does a very good job at classifying the German traffic signs. I look forward to potentially improving on the process in the future.