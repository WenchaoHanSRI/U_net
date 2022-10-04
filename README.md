# U_net for retinal fundus segmentation

The repository is specifically build for a task for personal pupose. The model is trained for semantic segmentation for retinal fundus images.
Feel free to play/modify the code for your own purpose.

- [Quick start](#quick-start)
  - [Data](#data)
  - [Without Docker](#without-docker) 
  - [With Docker](#with-docker)
- [Description](#description)
- [Usage](#usage)
  - [Docker](#docker)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)



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

3. Download the pre-trained model and put the model in the master directory:(https://figshare.com/ndownloader/files/34969398)


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


## Description


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
