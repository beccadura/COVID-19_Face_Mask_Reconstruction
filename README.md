# Face Mask Reconstruction

## Download
Training Raw Video: https://drive.google.com/uc?id=1smUOu5MzR_7khwz1sdJTw6MlehnZo4fj&export=download
Testing Raw Video: https://drive.google.com/uc?id=1vNBAJvIqAVAQkbIw-TvUffhMf1aifUlG&export=download

## CelebA data was prepared as follows:

  - Downloaded CelebA dataset then cropped the image to 512x512 pixels
  - Reduced size of dataset, as model takes a while to train
  - Created synthetic face mask and binary image on the cropped set with code adapted from [here](https://github.com/aqeelanwar/MaskTheFace) 
  
## Training steps:
- Train segmentation model in ***segmentation_trainer.py***
- Train inpainting model in ***reconstruction_trainer.py***
- Using ***infer.py*** with 2 checkpoints from above tasks to do inference

## Train facemask segmentation

```
python train.py segm
```

## Train facemask inpainting

```
python train.py facemask
```

## Paper References:
- Model based off the following paper: [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/abstract/document/9019697)

## Code References
- Folder structure and some code adapted from: https://github.com/kaylode/image-inpainting/tree/dff72fa655986f9b8776eb2df28ab8f3e06aa0f6
