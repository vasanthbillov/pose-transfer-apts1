

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
posetransfer
│
├───datasets
│   └───DeepFashion
│       ├───img
│       ├───img_seg
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
│               ├───pose_transfer_netG.pth
└───pose-transfer-apts1
```

### External Links
<h4>
  <a href="https://drive.google.com/drive/folders/11jM3r2kZHpO5O6TPOLsirz5W3XfPvZib">Dataset</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/file/d/148cg0x3SoJRqteqWiZoUARanyuTluv_q/view?usp=share_link">Pretrained Models</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
</h4>





