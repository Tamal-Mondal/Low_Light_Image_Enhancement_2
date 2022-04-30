"""Module that generate comparable results at folder data/compare/ where first image is the: 
    [Original Image] [Imagewithout Attention] [Image with Attention]

    """

import matplotlib.pyplot as plt
import imageio
import os


plt.rcParams.update({"figure.max_open_warning": 0})

original_path = "data/test_data/real"
without_attention_path = "data/result_Zero_DCE++/baseline"
with_attention_path = "data/result_Zero_DCE++/attention_no_bn_bias"

original = os.listdir(original_path)
with_attention = os.listdir(with_attention_path)
without_attention = os.listdir(without_attention_path)

NUM = len(original)


for num in range(NUM):
    
    print(original[num])
    
    plt.figure(figsize=(20, 17))

    img1 = imageio.imread(f"{original_path}/{original[num]}")
    img2 = imageio.imread(f"{without_attention_path}/{without_attention[num]}")
    img3 = imageio.imread(f"{with_attention_path}/{with_attention[num]}")

    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(img1)

    plt.subplot(1, 3, 2)
    plt.title("without_attention")
    plt.imshow(img2)

    plt.subplot(1, 3, 3)
    plt.title("attention_no_bn_bias")
    plt.imshow(img3)

    plt.savefig(f"data/compare/compare_{num}.jpg")
