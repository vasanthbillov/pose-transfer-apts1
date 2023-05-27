import numpy as np
import os
import pandas as pd
from pathlib import PureWindowsPath, PurePosixPath,Path

root_path = '/PoseTransfer_MS_RnD'
dataset_name = 'deepfashion/'

dataset_root = f'{root_path}/datasets/{dataset_name}'
train_csv = '/PoseTransfer_MS_RnD/datasets/deepfashion/train_img_pairs.csv'
test_csv = '/PoseTransfer_MS_RnD/datasets/deepfashion/test_img_pairs.csv'

def get_csv_path(strn):
    if strn == 't1':
        return train_csv
    else:
        return test_csv

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
                out_path=str(PurePosixPath(PureWindowsPath(out_path)))
                out_path =out_path.replace('D:\/', 'D:/')

                img_root= get_rel_path(out_path, 'img_seg')
                img_root = img_root.replace( 'img_seg', 'img')
                available_img_list.append(img_root)
    return  available_img_list       # print(img_root)


available_img_list = get_img_list()
# print(available_img_list)

img_train_df = pd.read_csv(get_csv_path('t1'))

img_train_df['imgA_check'] = img_train_df['imgA'].apply(lambda x: x in available_img_list)
img_train_df['imgB_check'] = img_train_df['imgB'].apply(lambda x: x in available_img_list)

# df1 = df[dimg_train_dff['imgA_check']==False][['img','label','edges','f_name','out_root']]
# print(df1.shape)
df= img_train_df[img_train_df['imgA_check']==True][['imgA','imgB']]

def convert_path(x):
    x = x.replace('jpg', 'png')
    x = x.replace('img', 'img_seg')
    return x

df['imgA_seg'] = df['imgA'].apply(convert_path)
df['imgB_seg'] = df['imgB'].apply(convert_path)

print(df)

df.to_csv(train_csv) 
