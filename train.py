import json
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from rich import print
import os
import time
import argparse
import logging
import uuid
import googlenetv1
import googlenetv3
import resnet50
import seaborn as sns

def normalize_ds(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


DATA_DIR = pathlib.Path("data")

batch_size = 64
img_height = 300
img_width = 300
EPOCHS = 0
ARGS = None


checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


CP_CALLBACK = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq="epoch",
    save_best_only=True,
)

ES_CALLBACK = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=5,
    start_from_epoch=10,
)

LOGGER = logging.getLogger(__name__)
ARGS = None

def prepare_dataset(img_height:int = 300, img_width:int = 300):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        color_mode="grayscale",
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        color_mode="grayscale",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    # Normalize Dataset to [0,1]
    train_ds = train_ds.map(normalize_ds)
    val_ds = val_ds.map(normalize_ds)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)
    
    return (train_ds, val_ds)

def train():
    if ARGS.type == "googlenetv1":
        if ARGS.no_augment:
            model = googlenetv1.prepare_model(channels=1)
        else:
            model = googlenetv1.prepare_model(no_augment=False, channels=1)
        model.summary()
        model.save_weights(checkpoint_path.format(epoch=0))

        ds_train, ds_val = prepare_dataset()
        # history = model.fit(x_train, [y_train, y_train, y_train], validation_data=(x_val, [y_val, y_val, y_val]), batch_size=64, epochs=40)
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            batch_size=64,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[CP_CALLBACK, ES_CALLBACK],
        )

    if ARGS.type == "resnet50":
        model = resnet50.prepare_model(channels=1)
        model.summary()
        model.save_weights(checkpoint_path.format(epoch=0))
        
        ds_train, ds_val = prepare_dataset()
        
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            batch_size=64,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[CP_CALLBACK, ES_CALLBACK],
        )
    
    if ARGS.type == "googlenetv3":
        model = googlenetv3.prepare_model(channels=1)
        model.summary()
        model.save_weights(checkpoint_path.format(epoch=0))
        
        ds_train, ds_val = prepare_dataset()
        
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            batch_size=64,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[CP_CALLBACK, ES_CALLBACK],
        )

    model.save(f"{checkpoint_dir}/model.h5")
    json.dump(history.history, open(f"{checkpoint_dir}/history.json", 'w'))

    df_data = pd.DataFrame.from_dict(history.history)
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))

    sns.lineplot(df_data[["loss", "val_loss"]], ax=axs[0])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')

    sns.lineplot(df_data[["accuracy", "val_accuracy"]] , ax=axs[1])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')

    fig.savefig(f"{checkpoint_dir}/visualization.png")
    
def dry_run():
    LOGGER.warning("Dry Run enabled")
    if ARGS.type == "googlenetv1":
        if ARGS.no_augment:
            model = googlenetv1.prepare_model()
        else:
            model = googlenetv1.prepare_model(no_augment=False, channels=1)
        model.summary()
        model.save_weights(checkpoint_path.format(epoch=0))
    if ARGS.type == "resnet50":
        model = resnet50.prepare_model()
        model.summary()
        model.save_weights(checkpoint_path.format(epoch=0))
    if ARGS.type == "googlenetv3":
        model = googlenetv3.prepare_model()
        model.summary()
        model.save_weights(checkpoint_path.format(epoch=0))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="How many epochs to train",default=40, type=int)
    parser.add_argument("-dir", "--directory", help="Directory of data", default="data", type=str)
    parser.add_argument("-t", "--type", help="Which type to train. Currently supportet: googlenetv1, resnet50)", default="googlenetv1", type=str)
    parser.add_argument("--no-augment", help="No augmentation steps (assumes augmentation is done beforehand)", action="store_true", default=False)
    parser.add_argument(
        "-d", "--dry",
        help="DryRun only - Will output model summary!",
        action="store_true",
    )
    ARGS = parser.parse_args()
    EPOCHS = ARGS.epochs
    DATA_DIR = pathlib.Path(ARGS.directory)

    if ARGS.dry:
        dry_run()
    else:
        train()

