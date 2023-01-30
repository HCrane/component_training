# MA_CNN_TRAINING

This repository contains allt he scripts needed to train a CNN with one of the 3 architectures GoogLeNetV1/V3 and Resnet50V2.

:warning: Be aware of early stopping and fine tune it to your needs! 


The following snippet shows, how a privet repository can be cloned automatically. Beforehand a Github Classic Personal Token needs to be created for the specific Repository to be able to download it without authentication.

```bash
#!/bin/bash

git clone https://"<insert token here>"0@github.com/"<insert path to GIT repo>"
mkdir data
screen -d -m aws s3 sync s3://"<name of S3 Bucket>"/data_resized data
```

The full script to execute in an EC2 instance with Tensorflow can be found in `user_data.sh`


```
usage: train.py [-h] [-e EPOCHS] [-dir DIRECTORY] [-t TYPE] [--no-augment] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        How many epochs to train
  -dir DIRECTORY, --directory DIRECTORY
                        Directory of data
  -t TYPE, --type TYPE  Which type to train. Currently supportet: googlenetv1, googlenetv3, resnet50)
  -d, --dry             DryRun only - Will output model summary!
```
