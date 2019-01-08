import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import label
import json as js
from shutil import copy2
import argparse
import sys
import gc 
from sklearn.model_selection import train_test_split

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def start(path_to_csv, output_dir):
   masks = pd.read_csv(path_to_csv)
   print(masks.shape[0], 'masks found')
   print(masks['ImageId'].value_counts().shape[0])
   masks.head()

   images_with_ship = masks.ImageId[masks.EncodedPixels.isnull()==False]
   images_with_ship = np.unique(images_with_ship.values)
   images_without_ship = masks.ImageId[masks.EncodedPixels.isnull()==True]
   images_without_ship = np.unique(images_without_ship.values)


   print('There are ' +str(len(images_with_ship)) + ' image files with masks')
   bboxes_dict = {}
   i = 0
   count_ships = 0

   for image in tqdm(images_with_ship):

    rle_0 = masks.query('ImageId=="'+image+'"')['EncodedPixels']
    mask_0 = masks_as_image(rle_0)
    lbl_0 = label(mask_0) 
    props = regionprops(lbl_0)
    bboxes = []
    count_ships = count_ships + len(props)
    for prop in props:
        bboxes.append(prop.bbox)
        
    i = i + 1
    if i % 500 == 0:
        gc.collect()    

    bboxes_dict[image] = bboxes.copy()

   bboxes_without_dict={}
    
   for image in tqdm(images_without_ship):
    
    bboxes = []
    bboxes.append([0,0,0,0])

    bboxes_without_dict[image] = bboxes.copy()

   i=0
   for j in tqdm(bboxes_dict):
    i=i+1
    fname,extension=j.split('.')
    lis=[]

    
    for k in bboxes_dict[j]:
        dist={'x1':k[1],
              'y1':k[0],
              'x2':k[3],
              'y2':k[2]}
        lis.append(dist)        
    dict={'rois':lis}
    json=js.dumps(dict)

    f=open(os.path.join(output_dir,fname+'.json'),'w')

    f.write(json)
    f.close
    if (1%1000==0):
        print(i,directory)

   i=0
   for j in tqdm(bboxes_without_dict):
    i=i+1
    fname,extension=j.split('.')
    lis=[]
    
    for k in bboxes_without_dict[j]:
        dist={'x1':k[1],
              'y1':k[0],
              'x2':k[3],
              'y2':k[2]}
        lis.append(dist)        
    dict={'rois':lis}
    json=js.dumps(dict)
    f=open(os.path.join(output_dir,fname+'.json'),'w')
    f.write(json)
    f.close
       
def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument("--path_to_csv", help="path to your csv file")
    parser.add_argument("--train_dir", help="path to your data folder")
    parser.add_argument("--eval_dir", help="path to your evals data folder")
    
    args = parser.parse_args()
    arguments = args.__dict__
    path_to_csv= arguments["path_to_csv"]
    train_dir= arguments["train_dir"]
    eval_dir= arguments["eval_dir"]
    start(path_to_csv,train_dir)
    img_filelist, roi_filelist = load_file_list(train_dir)
    Xtrain, Xtest, ytrain, ytest = train_test_split(img_filelist, labels, test_size=0.25, random_state=47)

    for i in tqdm(Xtest):
       basename=os.path.basename(i)
       os.rename(i,os.path.join(eval_dir,basename))
      
    for j in tqdm(ytest)):
       basename=os.path.basename(j)
       os.rename(i,os.path.join(eval_dir,basename))
   
    
    

if __name__ == '__main__':
     main(sys.argv)

