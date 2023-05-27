import numpy as np
import os
import pandas as pd
from pathlib import PureWindowsPath, PurePosixPath,Path


root_path = '/pose-transfer-repo'
dataset_name = 'deepfashion'

dataset_root = f'{root_path}/datasets/{dataset_name}'
img_pairs_train = f'{dataset_root}/train_img_pairs.csv'
img_pairs_test = f'{dataset_root}/test_img_pairs.csv'

img_pairs_train1 = f'{dataset_root}/train_img_pairs1.csv'
img_pairs_test1 = f'{dataset_root}/test_img_pairs1.csv'


def get_csv_path(strn):
    if strn == 'tn':
        return img_pairs_train
    else:
        return img_pairs_test

def path_replace(root1, label, file):
    label_src_root = root1.replace('img', label)
    return os.path.join(label_src_root,file)

def get_rel_path(root_path, idx_name): 
    # src = path_replace(root_path, idx_name, file)  
    path = Path(root_path)
    index = path.parts.index(idx_name)
    return  "/".join(path.parts[index:])

# Delete images in the img folder if its not in img_seg floder
def get_img_list():
    out_root = root_path + '/datasets/deepfashion/img_seg'
    available_img_list = []
    for (root,dirs,files) in os.walk(out_root, topdown=True):
            for file in files:
                file = file.replace('png', 'jpg')
                out_path = os.path.join(root, file)
                # out_path=str(PurePosixPath(PureWindowsPath(out_path)))
                # out_path =out_path.replace('D:\/', 'D:/')
                img_root=get_rel_path(out_path,'img_seg')
                img_root = img_root.replace('img_seg', 'img')
                available_img_list.append(img_root)
    return  available_img_list


available_img_list = get_img_list()

img_train_df = pd.read_csv(get_csv_path('ts'))
img_train_df['imgA_check'] = img_train_df['imgA'].apply(lambda x: x in available_img_list)
img_train_df['imgB_check'] = img_train_df['imgB'].apply(lambda x: x in available_img_list)

df = img_train_df[(img_train_df['imgA_check']==True) & (img_train_df['imgB_check']==True) ][['imgA','imgB']]

def convert_path(x):
    x = x.replace('jpg', 'png')
    x = x.replace('img', 'img_seg')
    return x

df['imgA_seg'] = df['imgA'].apply(convert_path)
df['imgB_seg'] = df['imgB'].apply(convert_path)



df.to_csv(img_pairs_test1)
