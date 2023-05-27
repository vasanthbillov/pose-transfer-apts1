

# pose-transfer-apts1
Repo for pose transfer research


### :zap: Getting Started
```bash
mkdir posetransfer
cd posetransfer
mkdir -p datasets/DeepFashion
mkdir -p output/DeepFashion/ckpt/pretrained
git clone git@github.com:vasanthbillov/pose-transfer-apts1.git
cd pose-transfer-apts1
pip install -r requirements.txt
```

### Code organization for training, testing and evaluation
* Download dataset files from [Google Drive](https://drive.google.com/drive/folders/11jM3r2kZHpO5O6TPOLsirz5W3XfPvZib) and extract into `datasets/DeepFashion` directory.
* Download pretrained checkpoints from [Google Drive](https://drive.google.com/file/d/148cg0x3SoJRqteqWiZoUARanyuTluv_q/view?usp=share_link) into `output/DeepFashion/ckpt/pretrained` directory.
```
pose2pose
│
├───datasets
│   └───DeepFashion
│       ├───img
│       ├───test_pose_maps
│       ├───train_pose_maps
│       ├───test_img_keypoints.csv
│       ├───test_img_list.csv
│       ├───test_img_pairs.csv
│       ├───train_img_keypoints.csv
│       ├───train_img_list.csv
│       └───train_img_pairs.csv
├───output
│   └───DeepFashion
│       └───ckpt
│           └───pretrained
│               ├───netD_257500.pth
│               ├───netD_260500.pth
│               ├───netG_257500.pth
│               └───netG_260500.pth
└───pose-transfer
```

> The precomputed keypoints and posemaps are estimated using the provided utility scripts in [pose-transfer/utils](https://github.com/prasunroy/pose-transfer/tree/main/utils).

### External Links
<h4>
  <a href="https://drive.google.com/drive/folders/11jM3r2kZHpO5O6TPOLsirz5W3XfPvZib">Dataset</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/file/d/148cg0x3SoJRqteqWiZoUARanyuTluv_q/view?usp=share_link">Pretrained Models</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
</h4>


### License
```
Copyright 2023 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```



