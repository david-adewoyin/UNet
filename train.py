import os
import logging
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from unet import UNet
from utils.utils import load_image_from_path, load_dataset_path

img_dir = Path("./data/images/")
masks_dir = Path("./data/masks/")
checkpoint_dir = Path("./checkpoints/")


#configuration can be overrided using using args
EPOCHS = 10
BUFFER_SIZE = 256
BATCH_SIZE = 8
IMG_HEIGHT = 128
IMG_WIDTH = 128
LR = 1e-3

OUTPUT_CLASSES = 23


def train_unet(
    unet=UNet,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    learning_rate=LR,
    val_size=0.1,
  
    save_checkpoint: bool = True,
):
    imgs_path, masks_path = load_dataset_path(img_dir, masks_dir=masks_dir)
    dataset = tf.data.Dataset.from_tensor_slices((imgs_path, masks_path))

    dataset = dataset.map(lambda img, mask: load_image_from_path(
        img, mask, img_size=(IMG_HEIGHT, IMG_WIDTH)),
                          num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).shuffle(BUFFER_SIZE).cache()
  

    dataset_size = len(imgs_path)
    val_batches = (dataset
               .take(round(val_size*dataset_size))
               .prefetch(buffer_size=tf.data.AUTOTUNE)
                )

    train_batches= (dataset
               .skip(round(val_size*dataset_size))
               .prefetch(buffer_size=tf.data.AUTOTUNE)
                ) 


    callbacks = [
        ReduceLROnPlateau(patience=3, verbose=1),
        ModelCheckpoint(checkpoint_dir, verbose=1, save_best_only=True)
    ]

    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        run_eagerly=True,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
    ''')

    # fiting the model
    history = unet.fit(
        train_batches,
        epochs=epochs,
        validation_data=val_batches,
        validation_steps = 8,
        callbacks=callbacks,
    )

    if save_checkpoint:
        path = os.path.join(checkpoint_dir)
        unet.save(path)


def get_args():
    parser = argparse.ArgumentParser(
        description='Trains the UNet model on images and mask')
    parser.add_argument('-e',
                        '--epochs',
                        dest='epochs',
                        default=EPOCHS,
                        type=int,
                        help='Number of Epochs')
    parser.add_argument('-b',
                        '--batch-size',
                        dest='batch_size',
                        default=BATCH_SIZE,
                        type=int,
                        help='Number of batches')
    parser.add_argument('-lr',
                        '--learning-rate',
                        dest='lr',
                        default=LR,
                        type=float,
                        help='Learning rate')
    parser.add_argument('-c',
                        '--classes',
                        dest='classes',
                        default=OUTPUT_CLASSES,
                        type=int,
                        help='Number of classes')
    parser.add_argument('-bfs',
                        '--buffer_size',
                        dest='buffer',
                        default=BUFFER_SIZE,
                        type=int,
                        help='Size of buffer')
    parser.add_argument(
        '-val',
        '--validation',
        dest='val',
        default=0.1,
        type=float,
        help='Percent of the data that is used as validation(0-1)')
    parser.add_argument('-f',
                        '--load',
                        dest='load',
                        type=str,
                        default=False,
                        help='Load a model weight from file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s :%(message)s')

    unet = UNet(num_classes=args.classes)
    if (args.load):
        try:
            unet.load_weights(args.load)
            logging.info(f"Model loaded from {args.load}")
        except:
            logging.info(f"Unable to load weights from {args.load}")

    try:
      train_unet(
            unet=unet,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            buffer_size=args.buffer,
            val_size=args.val,
        )
    except KeyboardInterrupt:
        unet.save('./.checkpoint-interrupted')
        logging.info("Saved Weight")
        raise
