"""
Module that generate comparable results at folder data/compare/ where first image is the: 
[Original Image] [Results1] [Results2]
"""

import matplotlib.pyplot as plt
import imageio
import os

plt.rcParams.update({"figure.max_open_warning": 0})

def compare_resuts(original_path, without_attention_path, with_attention_path):

    original = os.listdir(original_path)
    with_attention = os.listdir(with_attention_path)
    without_attention = os.listdir(without_attention_path)

    for num in range(len(original)):
        print(original[num])

        plt.figure(figsize=(20, 17))

        img1 = imageio.imread(f"{original_path}/{original[num]}")
        img2 = imageio.imread(
            f"{without_attention_path}/{without_attention[num]}")
        img3 = imageio.imread(f"{with_attention_path}/{with_attention[num]}")

        plt.subplot(1, 3, 1)
        plt.title("original")
        plt.imshow(img1)

        plt.subplot(1, 3, 2)
        plt.title("Baseline(Zero-DCE)")
        plt.imshow(img2)

        plt.subplot(1, 3, 3)
        plt.title("attention_last_layer")
        plt.imshow(img3)

        plt.savefig(f"data/compare/compare_{num}.jpg")


if __name__ == "__main__":

    original_path = "data/test_data/real"
    without_attention_path = "data/result_Zero_DCE++/baseline"
    with_attention_path = "data/result_Zero_DCE++/attention_last_layer"
    compare_resuts(original_path, without_attention_path, with_attention_path)
