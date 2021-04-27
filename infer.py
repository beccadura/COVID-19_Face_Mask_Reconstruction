import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
from PIL import Image
import cv2
from models import UNetSemantic, UNetSemantic2
import argparse
import glob
from losses import *
from metrics import *
import numpy as np
import moviepy.editor
import os

class Predictor():
    def __init__(self):
        self.device = torch.device('cpu')
        self.masking = UNetSemantic().to(self.device)
        self.masking.load_state_dict(torch.load('weights/model_segm_25_100.pth', map_location='cpu'))

        self.inpaint = UNetSemantic2().to(self.device)
        self.inpaint.load_state_dict(torch.load('weights/model_18_92.pth', map_location='cpu')['G'])
        self.inpaint.eval()
        self.criterion_ssim = SSIM(window_size = 11)
        self.criterion_dice = DiceLoss()
        self.img_size = 512

    def save_image(self, img_list, save_img_path, nrow):
        img_list  = [i.clone().cpu() for i in img_list]
        imgs = torch.stack(img_list, dim=1)
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_image(imgs, save_img_path, nrow = nrow)
        print(f"Save image to {save_img_path}")

    def predict(self, outpath='sample/results.png'):

        SSIM = []
        DICE = []
        SSIM_mask = []
        DICE_mask = []

        jpgFilenamesList = glob.glob('datasets/Dataset1-David-baseline/test/image/*.jpg')
        for idx,image in enumerate(jpgFilenamesList):
            image = image[-13:-4]
            outpath=f'results/results_{image}.png'
            image2 = 'datasets/Dataset1-David-baseline/test/image_masked/'+image
            img = cv2.imread(image2+'_masked.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            img = img.unsqueeze(0)

            image3 = 'datasets/Dataset1-David-baseline/test/image/'+image
            img_ori = cv2.imread(image3+'.jpg')
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_ori = cv2.resize(img_ori, (self.img_size, self.img_size))
            img_ori = torch.from_numpy(img_ori.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            img_ori = img_ori.unsqueeze(0)

            video_train = moviepy.editor.VideoFileClip("datasets/Dataset1-David-baseline/test/raw/hamlet_and_twisters_test.mp4")
            video_train = video_train.set_fps(1)
            num_frames_train = len(list(video_train.iter_frames()))

            train = moviepy.editor.AudioFileClip("datasets/Dataset1-David-baseline/test/audio/audio_hamlet_and_twisters_test.mp3")

            tot_frames = len(os.listdir("datasets/Dataset1-David-baseline/test/image"))
            sound_array = train.to_soundarray()
            extra = sound_array.shape[0]%num_frames_train
            if extra != 0:
                sound_array = sound_array[:-extra]
            chunk_size = sound_array.shape[0]//num_frames_train
            audio_train = sound_array.reshape((num_frames_train,chunk_size,2))
            audio_train = audio_train[0:tot_frames,:,:]

            # a = np.array((audio_train[idx,:,:], audio_train[idx,:,:], audio_train[idx,:,:], audio_train[idx,:,:]))
            # a = a.reshape(-1)
            # a = a[0:(512*512)]
            # audio = a.reshape(1,512,512)

            audio_py = audio_train[idx,:,:].reshape(-1)
            audio_py = torch.from_numpy(audio_py.astype(np.float32)).contiguous()
            conv = nn.Linear(audio_py.shape[0],self.img_size)
            conv_output = conv(audio_py)
            conv_output = np.array((conv_output.data))
            conv_output = [conv_output] * self.img_size
            conv_output = np.array((conv_output)).reshape(1,1,self.img_size,self.img_size)
                
            audio = torch.from_numpy(conv_output.astype(np.float32)).contiguous()
            
            with torch.no_grad():
                mask_outputs = self.masking(img)

                outputs = mask_outputs

                for idx,i in enumerate(outputs):
                    for idx2,i2 in enumerate(i):
                        for idx3,i3 in enumerate(i2):
                            for idx4,i4 in enumerate(i3.data):
                                if i4 >= 0.5:
                                    outputs[idx][idx2][idx3][idx4].data = torch.tensor(1)
                                else:
                                    outputs[idx][idx2][idx3][idx4].data = torch.tensor(0)

                kernel = np.ones((3,3),np.uint8)
                erosion = cv2.erode(np.float32(outputs[0][0].data),kernel,iterations = 1)
                dilation = cv2.dilate(erosion,kernel,iterations = 1)
                dilation = torch.from_numpy(dilation).contiguous()
                dilation = dilation.unsqueeze(0)
                dilation = dilation.unsqueeze(0)

                out = self.inpaint(img, dilation, audio)
                inpaint = img * (1 - dilation) + out * dilation
            masks = img * (1 - dilation) + dilation

            loss_ssim = self.criterion_ssim(inpaint, img_ori)
            # print(loss_ssim)
            SSIM.append(loss_ssim)
            loss_dice = 1 - self.criterion_dice(inpaint, img_ori)
            # print(loss_dice)
            DICE.append(loss_dice)

            loss_ssim_mask = self.criterion_ssim(out * dilation, img_ori * dilation)
            # print(loss_ssim_mask)
            SSIM_mask.append(loss_ssim_mask)
            loss_dice_mask = 1 - self.criterion_dice(out * dilation, img_ori * dilation)
            # print(loss_dice_mask)
            DICE_mask.append(loss_dice_mask)
            
            self.save_image([img, masks, inpaint, img_ori], outpath, nrow=4)

        print("DICE: ", np.mean(DICE))
        print("SSIM: ", np.mean(SSIM))
        print("DICE Mask: ", np.mean(DICE_mask))
        print("SSIM Mask: ", np.mean(SSIM_mask))

        


if __name__ == '__main__':
    model = Predictor()
    model.predict()