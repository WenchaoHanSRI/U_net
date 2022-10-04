# U_net for retinal fundus segmentation

The repository is specifically build for a task for personal pupose. The model is trained for semantic segmentation for retinal fundus images.
Feel free to play/modify the code for your own purpose.

- [Description](#description)
- [Quick start](#quick-start)
  - [Data](#data)
  - [Without Docker](#without-docker) 
  - [With Docker](#with-docker)
- [Train](#Train)

- [Usage](#usage)
  - [Docker](#docker)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)

## Description
This package is a U-net package for semantic segmentaiton for the retinal dataset. This code is modified from the git repository: (https://github.com/milesial/Pytorch-UNet)
The modifications details are below:
1. Fixed the bug in the dataloader to load the 24-bit mask properly. 
2. Added color autmentation on top of the original code for training. Specifically added color augmentation as the retinal fundus images varies largely in brightness, contrast, and hue. The rotation augmentation did not implemented yet. Please note: do not implment geometric augmentation directly in the transform funtion, this will not apply to masks. I will update this function later.
3. Added random corpping specifically for my purpose of training. The trained model can directly used for inferencing for the full size image. I need to train the model at full resolution as the vessel structure is very thin. The original code downsampled images, which does not apply to my case. The user can still use the scale parameter to downsample the image.
4. Reimplemented the validation function with four more error metrics. The predict.py function in the original package has bugs which I do not have time to fix. 
5. A note book file is created using google Colab. Please make sure you have more than 2G free space to run the code in Colab as it will automatically download the dataset to the clouse drive. 


## Quick start

### Data
The retinal fundas dataset can be downloaded:
(https://figshare.com/ndownloader/files/34969398)

The dataset includes 800 2048x2048 images in total: 600 training dataset and 200 testing dataset. 

### Without Docker

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Place the testing image under the master folder within
'/data/testimgsall' and corresponding masks in '/data/testmasksall' to perform segmentation. 
Two sample images are placed in the folder already, you can place more images of the downloaded data in the same folder.

3. Download the pre-trained model and put the model in the master directory:([https://figshare.com/ndownloader/files/34969398](https://drive.google.com/file/d/18jRivGhFium-URAhMMHTV-js-MsBUMzR/view?usp=sharing))


4. Download the data and run inferencing:
```bash
python validate.py
```
The user should expect a link print out from the command window, which brings to the Weights&Biases website and you can review your results there.


5. Alternatively, user can run without wandb module (i.e. Weights&Biases):
```bash
python validate_folder.py
```
The mean error metrics will be printed out in the command window, and the predicted mask image will save in the 'prediction' folder, which will be automatically created.

### With Docker
The user can also download the Docker image file to run the 'validation.py' file. The docker image file was tested on two Linux machines and have not yet tested on Windows machine. I will reease the updates to fix potential issues.
For users do not know how to use Docker image, I do not provide detailed instructions for now. However, this page will be updated to provide more support.

### Train
```bash
python train_cropsize.py -d=cuda:0 -cz=512 -scale=1 -e=15 -b=4
```
This is one example code that I used to train my model. If you do not have large enough GPU memory, you may reduce the batch size (e.g. -b=2), or you may reduce the crop size (e.g. -cz=256). However, the model may not have the same performance. 
The user can also use help arguement to check all the argments available for this training function.




## Usage
**Note : Use Python 3.6 or newer**

### Docker


### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

### Prediction


## Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```
Available scales are 0.5 and 1.0.




You can also download it using the helper script:

```
bash scripts/download_data.sh
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
