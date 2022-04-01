import cv2
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import shutil

def enhance_video_frame(data_lowlight, frame_no, result_path):
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	scale_factor = 12

	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
	enhanced_image,params_maps = DCE_net(data_lowlight)
	torchvision.utils.save_image(enhanced_image, result_path + str(frame_no) + ".png")
	return enhanced_image

def combine_images_to_video(images_path, video_path):

    images = [img for img in os.listdir(images_path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(images_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(images_path, image)))

    cv2.destroyAllWindows()
    video.release()

def enhance_video(video_path):
    
    original_video = cv2.VideoCapture(video_path)
    print("\nTotal number of frames in the video: ", int(original_video.get(cv2.CAP_PROP_FRAME_COUNT)), "\n")
    
    enhanced_images_path = "/home/cs19btech11056/cs21mtech12001-Tamal/DL_Project/Zero-DCE_extension/Zero-DCE++/data/result_Zero_DCE++/videos/temp/"
    enhanced_video_path = "/home/cs19btech11056/cs21mtech12001-Tamal/DL_Project/Zero-DCE_extension/Zero-DCE++/data/result_Zero_DCE++/videos/enhanced_video.avi"
    if os.path.exists(enhanced_images_path) and os.path.isdir(enhanced_images_path):
        shutil.rmtree(enhanced_images_path)
    os.mkdir(enhanced_images_path)

    success,image = original_video.read()
    no_of_frames = 1
    while success:   
        enhanced_image = enhance_video_frame(image, no_of_frames, enhanced_images_path)
        success,image = original_video.read()
        print("Processed frame: ", no_of_frames)
        if(no_of_frames == 100):
            break
        no_of_frames += 1
    
    print("\nPer frame image enhancement is done\n")
    combine_images_to_video(enhanced_images_path, enhanced_video_path)
    shutil.rmtree(enhanced_images_path)
    

if __name__ == '__main__':

	with torch.no_grad():
		videoFilePath = '/home/cs19btech11056/cs21mtech12001-Tamal/DL_Project/Zero-DCE_extension/Zero-DCE++/data/test_data/videos/pexels-gabriel-peregrino-5686177.mp4'	
		enhance_video(videoFilePath)
		