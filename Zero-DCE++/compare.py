"""
Module that generate comparable results at folder data/compare/ where first image is the: 
[Original Image] [Results1] [Results2]
"""

import matplotlib.pyplot as plt
import imageio
import os

plt.rcParams.update({"figure.max_open_warning": 0})

def compare_resuts(original_path, baseline_path, with_extra_iterations_path, with_attention_path, with_attention_no_bn_bias_path, \
        with_attention_pool_no_bn_bias_path, with_attention_last_layer_path, with_attention_reduced_rr_path):

    original = os.listdir(original_path)

    for num in range(len(original)):
        print(original[num])

        fig = plt.figure(figsize=(20, 17))
        rows, columns = 4, 4

        img1 = imageio.imread(f"{original_path}/{original[num]}")
        img2 = imageio.imread(f"{baseline_path}/{original[num]}")
        img3 = imageio.imread(f"{with_extra_iterations_path}/{original[num]}")
        img4 = imageio.imread(f"{with_attention_path}/{original[num]}")
        img5 = imageio.imread(f"{with_attention_no_bn_bias_path}/{original[num]}")
        img6 = imageio.imread(f"{with_attention_pool_no_bn_bias_path}/{original[num]}")
        img7 = imageio.imread(f"{with_attention_last_layer_path}/{original[num]}")
        img8 = imageio.imread(f"{with_attention_reduced_rr_path}/{original[num]}")
        
        fig.add_subplot(rows, columns, 1)
        plt.axis("off")
        plt.title("Original")
        plt.imshow(img1)

        fig.add_subplot(rows, columns, 2)
        plt.axis("off")
        plt.title("Baseline(Zero-DCE)")
        plt.imshow(img2)

        fig.add_subplot(rows, columns, 3)
        plt.axis("off")
        plt.title("With extra 4 iterations")
        plt.imshow(img3)
        
        fig.add_subplot(rows, columns, 4)
        plt.axis("off")
        plt.title("With attention in first 6 layers")
        plt.imshow(img4)

        fig.add_subplot(rows, columns, 5)
        plt.axis("off")
        plt.title("With attention, no Batchnorm and bias in Conv layers")
        plt.imshow(img5)

        fig.add_subplot(rows, columns, 6)
        plt.axis("off")
        plt.title("With all 4 pooling in channel attention")
        plt.imshow(img6)
        
        fig.add_subplot(rows, columns, 7)
        plt.axis("off")
        plt.title("With attention in all layers")
        plt.imshow(img7)

        fig.add_subplot(rows, columns, 8)
        plt.axis("off")
        plt.title("With reduced reduction_rate in attention")
        plt.imshow(img8)

        plt.savefig(f"data/compare/compare_{num}.jpg")


if __name__ == "__main__":

    original_path = "data/test_data/real"
    baseline_path = "data/result_Zero_DCE++/baseline"
    with_extra_iterations_path = "data/result_Zero_DCE++/extra_iterations"
    with_attention_path = "data/result_Zero_DCE++/attention"
    with_attention_no_bn_bias_path = "data/result_Zero_DCE++/attention_no_bn_bias"
    with_attention_pool_no_bn_bias_path = "data/result_Zero_DCE++/attention_pool_no_bn_bias"
    with_attention_last_layer_path = "data/result_Zero_DCE++/attention_last_layer"
    with_attention_reduced_rr_path = "data/result_Zero_DCE++/attention_reduced_rr"
    compare_resuts(original_path, baseline_path, with_extra_iterations_path, with_attention_path, with_attention_no_bn_bias_path, \
        with_attention_pool_no_bn_bias_path, with_attention_last_layer_path, with_attention_reduced_rr_path)
