import cv2
import time
#import brisque
from libsvm import svmutil

#brisq = brisque.BRISQUE()
#print(brisq.get_score("data/result_Zero_DCE++/baseline/iith-2.jpg"))

import imquality.brisque as brisque
import PIL.Image

path = 'data/result_Zero_DCE++/attention_4_layers/iith-2.jpg'
img = PIL.Image.open(path)
time_start = time.time()
print(brisque.score(img))
time_used = time.time() - time_start
print(f'time used in sec: {time_used:.4f}')