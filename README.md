## Implementation of UNet Image Segmentation using Tensorflow 

The architecture was based on the original paper [U-Net:Convolutional Network for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597)   

### Overview
The model was trained on the CARLA image segmentation dataset, and can be found in the data folder.  


### Network Architecture
The Architecture was based on the original Unet Paper which consists of a contracting path which downsamples the image size and a corresponding expanding path. You can see how the model was implemented by going through the repository or view the corresponding [notebook](https://github.com/david-adewoyin/UNet/blob/main/unet.ipynb) in the repo.

 <img width="500" alt="unet" src="https://user-images.githubusercontent.com/57121852/196273918-208c1a81-4387-4a84-a174-cd5b2100a4a9.png">


### Quick start
Install Dependencies
```
pip install -r requirements.txt
```

### Training
You can try the model on your custom data by coping it into the data folder or changing the path in  ```train.py``` 
```
> python train.py --help

usage: train.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-lr LR] [-c CLASSES]
                [-bfs BUFFER] [-val VAL] [-f LOAD]

Trains the UNet model on images and mask

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of Epochs
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of batches
  -lr LR, --learning-rate LR
                        Learning rate
  -c CLASSES, --classes CLASSES
                        Number of classes
  -bfs BUFFER, --buffer_size BUFFER
                        Size of buffer
  -val VAL, --validation VAL
                        Percent of the data that is used as validation(0-1)
  -f LOAD, --load LOAD  Load a model weight from file
```
### Prediction
After training your model and saving it to the `checkpoint`, you can easily test the output mask on your images via the CLI

To predict a single image and save it:
```
python predict.py -i image.jpg -o output.jpg
```

``` 
> python predict.py --help

usage: predict.py [-h] [-m MODEL] -i INPUT [INPUT ...]
                  [-o OUTPUT [OUTPUT ...]]

Predict masks from input image

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        specify where the model path is save
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Filenames of input images
  -o OUTPUT [OUTPUT ...], --output OUTPUT [OUTPUT ...]
                        Filenames of output Images
```
