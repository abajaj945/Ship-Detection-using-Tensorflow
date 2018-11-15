# Ship-Detection-using-Tensorflow
Using the Airbus ship detection data of satellite images I have created a ship detection model using Squeezenet and Yolo algorithm.

## Training 
I trained the model on google cloud for 3 days untill I burned all my cloud credits. The model achieved an an IOU of 0.442 over train dataset. The model did not performed well on test-dataset which I believe is due to less training hours and another reason as cited by writers of YOLO paper (https://arxiv.org/pdf/1506.02640.pdf) "the algorithm struggles with small objects that appear in groups, such as flocks of birds".The model can be improved by

using Squeezenet17,
better datasampling,
more training hours
