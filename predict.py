import os
import logging
import argparse
import tensorflow as tf
from pathlib import Path
from unet import UNet
from utils.utils import create_mask


def predict_image(unet, img):
    mask = unet(img)
    mask = create_mask(mask)
    return mask


def load_image(img_path, img_size=(128, 128)):
    img_data = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_data)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size, method='nearest')
    img = tf.expand_dims(img, axis=0)
    return img


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks from input image")
    parser.add_argument('-m',
                        '--model',
                        dest='model',
                        default='./checkpoints',
                        help='specify where the model path is save')
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        help='Filenames of input images',
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        metavar='OUTPUT',
                        nargs='+',
                        help='Filenames of output Images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = args.output

    unet = tf.keras.models.load_model(args.model)

    for i, filename in enumerate(in_files):
        img = load_image(filename)
        mask = predict_image(unet=unet, img=img)
        predicted_img = tf.keras.preprocessing.image.array_to_img(mask)

        outfile = out_files[i]
        predicted_img.save(outfile)
        logging.info('Mask saved to out_file {outfile}')
