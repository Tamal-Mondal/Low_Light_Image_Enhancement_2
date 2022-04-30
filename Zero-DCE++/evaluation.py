from math import log10, sqrt
import cv2
import glob
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error

def scale_image(data_lowlight):
    scale_factor = 12
    data_lowlight = (data_lowlight/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    
    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    return data_lowlight.cpu().detach().numpy()
 
def calculate_PSNR(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_SSIM(original, enhanced):
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(gray_original, gray_enhanced, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def main():
    actual_images_folder = "/home/cs19btech11056/cs21mtech12001-Tamal/DL_Project/Zero-DCE_extension/Zero-DCE++/data/test_data/real"
    enhanced_images_folder = "/home/cs19btech11056/cs21mtech12001-Tamal/DL_Project/Zero-DCE_extension/Zero-DCE++/data/result_Zero_DCE++/extra_iterations"
    
    actual_images = glob.glob(actual_images_folder + "/*") 
    enhanced_images = glob.glob(enhanced_images_folder + "/*") 
    
    total_psnr, total_SSIM_score, total_mae = 0, 0, 0
    for i in range(len(actual_images)):
        print(actual_images[i])
        original = scale_image(cv2.imread(actual_images[i]))
        enhanced = cv2.imread(enhanced_images[i], 1)
        
        total_psnr += calculate_PSNR(original, enhanced)
        
        ssim_score, ssim_diff = calculate_SSIM(original, enhanced)
        total_SSIM_score += ssim_score
        
        total_mae += mean_absolute_error(original.flatten(), enhanced.flatten())
        
    print("Average PSNR value is {} dB".format(total_psnr/len(actual_images)))
    print("Average SSIM score is {}".format(total_SSIM_score/len(actual_images)))
    print("Average MAE value is {}".format(total_mae/len(actual_images)))
 
if __name__ == "__main__":
	main()
