# Face Mask Reconstruction (Proposed Model)
## Authors: Becca Dura, David Kinney, Linda Kopecky, & Evgeni Radichev

## Download
- Training Raw Video (needs to be placed in /datasets/Dataset1-David-baseline/train/raw): https://drive.google.com/uc?id=1smUOu5MzR_7khwz1sdJTw6MlehnZo4fj&export=download
- Testing Raw Video (needs to be placed in /datasets/Dataset1-David-baseline/test/raw): https://drive.google.com/uc?id=1vNBAJvIqAVAQkbIw-TvUffhMf1aifUlG&export=download

## celeba data was prepared as follows:

  - Downloaded CelebA dataset (https://www.kaggle.com/jessicali9530/celeba-dataset) then cropped the image to 512x512 pixels
  - Reduced size of dataset, as model takes a while to train
  - Created synthetic face mask and binary image on the cropped set with code adapted from: https://github.com/aqeelanwar/MaskTheFace

## Dataset1-David-baseline data was prepared as follows:

  - Split training and testing videos into images (1 frame per second)
  - Extracted audio from videos
  - Reduced size of training dataset, as model takes a while to train
  - Cropped images to be 512x512 pixels
  - Created synthetic face mask and binary image on the cropped set with code adapted from: https://github.com/aqeelanwar/MaskTheFace
  
## Training Steps:
- Download raw videos from links above (they are too large to be pushed to GitHub) and place them in correct file locations.
- Train segmentation model in ***segmentation_trainer.py***.
- Train reconstruction model in ***reconstruction_trainer.py***.
- Use ***infer.py*** with final weights files from training the segmentation and reconstruction models to make predictions.
- If you just want to output the reconstructed images, use ***infer_jpgs.py*** instead of ***infer.py***.

## Train Face Mask Segmentation Model

```
python3 train.py segm
```

## Train Face Mask Reconstruction Model

```
python3 train.py facemask
```


## Predict Image Outputs 
#### (prints comparison with masked image, binary image overlayed over masked image, reconstructed image, & ground truth)

```
python3 infer.py
```

## Predict Image Outputs 
#### (prints jpg of reconstructed image only)

```
python3 infer_jpgs.py
```

## File Summaries

- Train.py: used to determine whether we are running the segmentation or the reconstruction module.
- Segmentation_trainer.py: trains the segmentation module and prints out metrics during the training process.  Weights for this module are saved at each epoch.
- Reconstruction_trainer.py: trains the reconstruction module and prints out metrics during the training process.  Visuals are also printed for each epoch so we can see how the images in the batches are being generated and how the reconstructed images compare to the ground truth images.  Weights for this module are saved at each epoch.
- Infer.py: reconstructs images in the testing dataset.  This runs the testing data through the trained segmentation and reconstruction modules and reconstructs the face by using the face mask region of the generated image over the initial masked image.  This script also performs erosion and dilation after the segmentation module to ensure the binary segmentation image is clean and does not contain any noise outside the masked region.  One thing to note about this file is that the final weights files from training the modules must be correctly specified in this file for it to work properly.  This file outputs the DICE and SSIM metrics as well as a png file for each frame containing the masked image, segmented mask image, reconstructed image, and ground truth image.
- Infer_jpgs.py: Does the same thing as infer.py, but it outputs only a jpg of the reconstructed image.
- Datasets folder: contains the datasets used for training and testing the model as well as the MaskTheFace code used to add synthetic masks onto the unmasked video frame images.
- Weights folder: contains the weights for each model.  Note: our final weights files were significantly too large to upload to GitHub or Google Drive, so there are not currently any weights files to run the code with.
- Models folder: contains the code for the actual models used in each module (i.e. the segmentation and reconstruction GANS, which are similar to U-Nets, the discriminators, and the pre-trained VGG-19 perceptual network).
- Configs folder: contains the configuration files specified when training the models.  These files specify the location of the data, the number of epochs, the batch size, the learning rates, etc…
- Metrics folder: contains the code for the metrics used to evaluate the model.
- Losses folder: contains the code for the loss functions used when training the models.
- Sample folder: contains the visuals produced when training the reconstruction model.  Note that not all the samples from when we trained the model have been uploaded to this folder.
- Results folder: contains the results when reconstructing the images from the training and testing sets using the trained model.


## Paper References:
- Model based off the following paper: [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/abstract/document/9019697)

## Code References
- Folder structure and some code adapted from: https://github.com/kaylode/image-inpainting/tree/dff72fa655986f9b8776eb2df28ab8f3e06aa0f6
