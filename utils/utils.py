import tensorflow as tf
import matplotlib.pyplot as plt
import os 
from os.path import splitext




def load_dataset_path(images_dir: str,masks_dir: str,  mask_suffix: str = ''):
    """
    Load the training dataset(images,masks) from the given path into tf.data.Dataset
    """

    # load the filenames into imgs
    imgs = [file for file in os.listdir(images_dir) if not file.startswith(".")]
    print(imgs[:10])
    if not imgs:
        raise RuntimeError(
            f"No input file found in {images_dir} make sure you put your images here"
        )
    # load the masks into the variable, expect the masks to have the same name as the images
    masks = [os.path.join(masks_dir,splitext(img)[0]) + mask_suffix + '.png' for img in imgs] 
    imgs = [os.path.join(images_dir,img) for img in imgs]

    return imgs,masks
     
     

@tf.function
def load_image_from_path(img_path, mask_path,img_size=(256,256)):
    """
    Load the given image and mask from the path
    """
    img_data = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_data,channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)

    segm_data = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(segm_data)
    mask = tf.math.reduce_max(mask, axis = -1, keepdims=True)

    img = tf.image.resize(img, img_size , method = 'nearest')
    mask = tf.image.resize(mask,img_size, method = 'nearest')
  
    return img, mask


@tf.function
def augment(image,mask, seed):
    image = tf.image.random_crop(image, (256, 256, 3),seed = seed)
    mask  = tf.image.random_crop(mask,(256,256,3),seed = seed)
    return image,mask


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis = -1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]


""" def show_predictions(dataset=None, num=1):
    
    Displays the first image of each of the num batches
    
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
             create_mask(unet.predict(sample_image[tf.newaxis, ...]))]) """
             
def display_img_and_mask(display_list=[]):
    """
    plot a list of images in a grid
    Args:
        display_list: list of images vector 
    """
   
    plt.figure(figsize=(10,10))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
      plt.axis('off')
      plt.show()