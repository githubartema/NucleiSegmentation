# Finding the nuclei in divergent images to advance medical discovery

Implementation of UNet in Keras, weights for the trained model are included.
The performance of an algorithm is evaluated with Dice coefficient (0.87 on the validation set). 

# Usage
## Evaluation
There is a `Predict_masks.py` script which can be used to evaluate the model and predict masks for the test dataset. The weights are stored in the "weights" directory.

It is necessary to point the directory where the test dataset is stored. Predicted masks will be stored in the same directories as the test images.

Usage example: 
``` 
python3 Predict_masks.py -dir /Users/user/Documents/nuclei_segmentation/data/input/stage1_test/ 
``` 
### Arguments: 
```
-dir    : Pass the full path of a directory containing a set of test images.
``` 
The hierarchy of folders must be the same as it comes from Kaggle zip (each photo is stored in its own folder)

## Training
The model is supposed to be trained on the dataset from the Kaggle competition. For the pretrained model I used "stage1_train.zip" which contains roughly 670 images and their masks.

It is necessary to point the directory where the train dataset is stored.

Usage example: 
``` 
python3 Train.py -dir /Users/user/Documents/nuclei_segmentation/data/input/stage1_train/ 
``` 
### Arguments: 
```
-dir    : Pass the full path of a directory containing a set of train images.
```

The hierarchy of folders must be the same as it comes from Kaggle zip (each photo is stored in its own folder)

# Requirements
- Keras
- Tensorflow 
- Numpy
- Path.py
- OpenCV
