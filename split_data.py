import glob
from sklearn.model_selection import train_test_split
import os


def extract_filename_without_extension(filename):
    basename = os.path.basename(filename)
    barename, extension = os.path.splitext(basename)
    return barename, filename



def load_file_list(directory):
    # load images, load jsons, associate them by name, XYZ.jpg with XYZ.json
    img_files1 = glob.glob(directory + "/*.jpg")
    img_files2 = glob.glob(directory + "/*.jpeg")
    img_files  = img_files1 + img_files2
    roi_files  = glob.glob(directory + "/*.json")
    img_kv = list(map(extract_filename_without_extension, img_files))
    roi_kv = list(map(extract_filename_without_extension, roi_files))
    all_kv = img_kv + roi_kv
    img_dict = dict(img_kv)
    roi_dict = dict(roi_kv)
    all_dict = dict(all_kv)
    outer_join = [(img_dict[k] if k in img_dict else None,
                   roi_dict[k] if k in roi_dict else None) for k in all_dict]
    # keep only those where the jpg and the json are both available
    inner_join = list(filter(lambda e: e[0] is not None and e[1] is not None, outer_join))
    if len(inner_join) == 0:
        return [], []
    else:
        img_list, roi_list = zip(*inner_join)  # unzip, results are a tuple of img names and a tuple of roi names
        return list(img_list), list(roi_list)   

def main():

   img_filelist, roi_filelist = load_file_list(directory)
   Xtrain, Xtest, ytrain, ytest = train_test_split(img_filelist, labels, test_size=0.25, random_state=47)
   for i in tqdm(Xtest):
       basename=os.path.basename(i)
       os.rename(i,os.path.join(eval_directory,basename))
      
   for j in tqdm(ytest)):
       basename=os.path.basename(j)
       os.rename(i,os.path.join(eval_directory,basename))
   
    
    
