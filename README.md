# Face Mask Reconstruction
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


## Predict Image Outputs (prints comparison with masked image, binary image overlayed over masked image, reconstructed image, & ground truth)

```
python3 infer.py
```

## Predict Image Outputs (prints jpg of reconstructed image only)

```
python3 infer_jpgs.py
```

## Paper References:
- Model based off the following paper: [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/abstract/document/9019697)

## Code References
- Folder structure and some code adapted from: https://github.com/kaylode/image-inpainting/tree/dff72fa655986f9b8776eb2df28ab8f3e06aa0f6
