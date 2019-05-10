# Ship-Detection-using-Tensorflow
Using the Airbus data of satellite images I have created a ship detection model using Squeezenet and Yolo algorithm.

## Training 
I trained the model on google cloud for 2 days untill I burned all my cloud credits. The model achieved an an IOU of 0.442 over test dataset. The model did not performed very-good on test-dataset because of less training hours and another reason as cited by writers of YOLO paper (https://arxiv.org/pdf/1506.02640.pdf) "the algorithm struggles with small objects that appear in groups, such as flocks of birds".The model can be improved by,

1. using Squeezenet17,
2. better datasampling(only 40000 images had ships in 200000 images),
3. more training hours


## Usage Instructions
```
git clone https://github.com/abajaj945/Ship-Detection-using-Tensorflow.git
cd Ship-Detection-using-Tensorflow
```
Download the ship data from kaggle https://www.kaggle.com/c/airbus-ship-detection, unzip it i will have a train_ship_segmentations_v2.csv file and and a folder containing training images. Create a new directory for evaluation data
Now prepare data for training purpose.

```
python3 generate_data.py --path_to_csv path/to/train_ship_segmentations_v2.csv --train_dir path/to/training images directory --eval-dir path/to/evaluation_dir
```

The data is split into training directory and evaluation directory


Now for training 

```
main.py --train_dir path/to/training images directory
```
