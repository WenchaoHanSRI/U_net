# U_net for retinal fundus segmentation

The repository is specifically built for a task for personal purpose. The model is trained for semantic segmentation for retinal fundus images.
Feel free to play/modify the code for your own purpose.

- [Quick-start](#quick-start)
- [Data](#data)
- [Description](#description)
- [Validation](#validation)
  - [Pretrained model](#pretrained-model)
  - [Without Docker](#without-docker)
  - [Weights-&-Biases](#Weights-&-Biases)
  - [With Docker](#with-docker)
- [Train](#Train)


## Quick-start
Using the two notebook files as a demo to 

1) inspect the data: 
https://github.com/WenchaoHanSRI/U_net/blob/main/Inspect_dataset.ipynb. 

2) using pre-trained weights to perform the prediction and evaluate the model performance: https://github.com/WenchaoHanSRI/U_net/blob/main/Validation.ipynb. 

NOTE: for validation, the notebook only includes 2 image samples. The user may upload more samples in the directory where the samples save in the Colab to run the experiments.

## Data
The retinal fundus dataset can be downloaded:
(https://figshare.com/ndownloader/files/34969398)
The dataset includes 800 2048x2048 images in total: 600 training dataset and 200 testing dataset.

## Description
This package is a U-net package for semantic segmentation for the retinal dataset. This code is built from a public U-net framework. 

The modifications details are below:
1. Fixed the bug in the dataloader to load both the 24-bit and 8-bit mask images properly. 
2. Added color augmentation in the function as the retinal fundus images varies largely in brightness, contrast, and hue. The rotation augmentation was not implemented yet. Please note: do not implement geometric augmentation directly using the transform function in the code, this will not apply to masks. I will update this function later.
3. Added random cropping specifically for my purpose of training. The trained model can be directly used for inferencing for the full size image. I need to train the model at full resolution as the vessel structure is very thin. The public U-net code downsampled images and masks before being fed into the model, which can result in missing thin vessel structures in both images and masks. The user can still use the scale parameter to downsample the image before being fed into the network if your images/masks do not have very small structures that may disappear whnen downsampling and upsampling.
4. Reimplemented the validation function with four more error metrics. The predict.py function in the original package has bugs which I do not have time to fix. 
5. A notebook file is created using google Colab. Please make sure you have more than 2G free space to run the code in Colab as it will automatically download the dataset to the clouse drive. 


## Validation


### Pretrained-model

(https://drive.google.com/file/d/18jRivGhFium-URAhMMHTV-js-MsBUMzR/view?usp=sharing)

### Without Docker

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Place the testing image under the master folder within
'/data/testimgsall' and corresponding masks in '/data/testmasksall' to perform segmentation. 
Two sample images are placed in the folder already, you can place more images of the downloaded data in the same folder.

3. Download the pre-trained model and put the model in the master directory:(https://drive.google.com/file/d/18jRivGhFium-URAhMMHTV-js-MsBUMzR/view?usp=sharing)


4. Download the data and run inferencing:
```bash
python validate.py
```

### Weights-&-Biases
The user should expect a link print out from the command window, which brings to the Weights&Biases website and you can review your results there.


5. Alternatively, user can run without wandb module (i.e. Weights&Biases):
```bash
python validate_folder.py
```
The mean error metrics will be printed out in the command window, and the predicted mask image will save in the 'prediction' folder, which will be automatically created.

### With Docker
The docker file can be downloaded:
(https://drive.google.com/file/d/1zbAVBKpBfNLdBN1amB066UXTloDjqvBF/view?usp=sharing) new version not yet tested;

(https://drive.google.com/file/d/1GEaAb3H6Wl-bcTD5XRjDA6SbGOSvY-3e/view?usp=sharing) old version and tested.

1. load the docker file:
```console
docker load < wenchao_docker_final.tar.xz
```
2. mount the directory that holds the test image files in the same format as my data directory.
example code:
```console
docker run -it -rm -v c:/Unet_project/:/home/user/U-net/data wenchao_final bash
```
In this example, your path for the test images are 'c:/Unet_project/'. User may change path based on where the test image files are saved.

3. get in the container:
```console
docker run --rm -it --entrypoint "/bin/bash" --memory=30g --shm-size=30g --memory-swap=15g  -v c:/Unet_project/:/home/user/U-Net/data wenchao_final
```

4. run the python file:
```bash
python validate.py
```

The user can also download the Docker image file to run the 'validation.py' file. The docker image file was tested on two Linux machines and have not yet tested on Windows machine. I will reease the updates to fix potential issues.
For users do not know how to use Docker image, I do not provide detailed instructions for now. However, this page will be updated to provide more support.

### Train
You can directly run the code below as it uses the data samples in '/data/imgs/' and '/data/masks/', which includes 3 samples. To train the model, please use the full training dataset and place the images and masks in the above mentioned folders without changing the folder name.

```bash
python train_cropsize.py -d=cuda:0 -cz=512 -scale=1 -e=15 -b=4
```
This is one example code that I used to train my model. If you do not have large enough GPU memory, you may reduce the batch size (e.g. -b=2), or you may reduce the crop size (e.g. -cz=256). However, the model may not have the same performance. 
The user can also use help arguement to check all the argments available for this training function.

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
