import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PoseDataset(Dataset):
    
    def __init__(self, dataset_dir, img_pairs, pose_maps_dir,
                 img_transform=None, map_transform=None, reverse=False):
        super(PoseDataset, self).__init__()
        self._dataset_dir = dataset_dir
        self._img_pairs = pd.read_csv(img_pairs)
        self._pose_maps_dir = pose_maps_dir
        self._img_transform = img_transform or transforms.ToTensor()
        self._map_transform = map_transform or transforms.ToTensor()
        self._reverse = reverse
    
    def __len__(self):
        return len(self._img_pairs)
    
    def __getitem__(self, index):
        pthA = self._img_pairs.iloc[index].imgA
        pthB = self._img_pairs.iloc[index].imgB

        pthA_seg = self._img_pairs.iloc[index].imgA_seg
        pthB_seg = self._img_pairs.iloc[index].imgB_seg
       
        fidA = os.path.splitext(pthA)[0].replace('/', '').replace('\\', '')
        fidB = os.path.splitext(pthB)[0].replace('/', '').replace('\\', '')
        # print(fidA, fidB)

        # print(os.path.join(self._dataset_dir+'/', pthA))
        imgA = Image.open(os.path.join(self._dataset_dir+'/', pthA))
        imgB = Image.open(os.path.join(self._dataset_dir+'/', pthB))

        imgA_seg = Image.open(os.path.join(self._dataset_dir+'/', pthA_seg)).convert('RGB')
        imgB_seg = Image.open(os.path.join(self._dataset_dir+'/', pthB_seg)).convert('RGB')

        # print('MAPA PATH:  ',self._pose_maps_dir, self._pose_maps_dir, (f'{fidA}.npz'))
        mapA = np.float32(np.load(os.path.join(self._pose_maps_dir+'/', f'{fidA}.npz'))['arr_0'])
        mapB = np.float32(np.load(os.path.join(self._pose_maps_dir+'/', f'{fidB}.npz'))['arr_0'])

        # print(mapA, mapB)
        
        imgA = self._img_transform(imgA)
        imgB = self._img_transform(imgB)

        imgA_seg = self._img_transform(imgA_seg)
        imgB_seg = self._img_transform(imgB_seg)
        
        mapA = self._map_transform(mapA)
        mapB = self._map_transform(mapB)
        
        if not self._reverse:
            return {'imgA': imgA, 'imgB': imgB,'imgA_seg': imgA_seg, 'imgB_seg': imgB_seg,  'mapA': mapA, 'mapB': mapB, 'fidA': fidA, 'fidB': fidB}
        else:
            return {'imgA': imgB, 'imgB': imgA,'imgA_seg': imgB_seg, 'imgB_seg': imgA_seg, 'mapA': mapB, 'mapB': mapA, 'fidA': fidB, 'fidB': fidA}


def create_dataloader(dataset_dir, img_pairs, pose_maps_dir, img_transform=None, map_transform=None, reverse=False, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):

    dataset = PoseDataset(dataset_dir, img_pairs, pose_maps_dir, img_transform, map_transform, reverse)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)


if __name__=='__main__':
    import torch


    root_path = '/home/ec2-user/SageMaker'
    dataset_name = 'deepfashion'

    dataset_root = f'{root_path}/datasets/{dataset_name}'
    img_pairs_train = f'{dataset_root}/train_img_pairs1.csv'
    img_pairs_test = f'{dataset_root}/test_img_pairs1.csv'
    pose_maps_dir_train = f'{dataset_root}/train_pose_maps'
    pose_maps_dir_test = f'{dataset_root}/test_pose_maps'

    dataset = PoseDataset(dataset_root, img_pairs_train, pose_maps_dir_train,
                                     img_transform = None, map_transform = None,
                                    reverse=False)
    
    # print('data: ',dataset.__getitem__(1))
    d1 = next(iter(dataset))
    print('imagA: ',d1['imgA'].shape)
    print('imagB: ',d1['imgB'].shape)
    print('imgA_seg: ',d1['imgA_seg'].shape)
    print('imgB_seg: ',d1['imgB_seg'].shape)
    segmab =torch.cat((d1['imgA_seg'], d1['imgB_seg']), dim=0)
    print('segmab shape: ',segmab.shape)
    print('mapA: ',d1['mapA'].shape)
    print('mapB: ',d1['mapB'].shape)
    mapab =torch.cat((d1['mapA'], d1['mapB']), dim=1)
    print('mapab shape: ',mapab.shape)
    print('fidA: ',d1['fidA'])
    print('fidB: ',d1['fidB'])    
  

