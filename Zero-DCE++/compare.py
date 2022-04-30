"""Module that generate comparable results at folder data/compare/ where first image is the: 
    [Original Image] [Results1] [Results2]

    """

import matplotlib.pyplot as plt
import imageio
import os


plt.rcParams.update({"figure.max_open_warning": 0})


def compare_resuts(baseline_path, results1_path, results2_path):


    baseline = os.listdir(baseline_path)
    resut1 = os.listdir(results1_path)
    results2 = os.listdir(results2_path)

    NUM = len(baseline)

    for num in range(NUM):
        plt.figure(figsize=(20, 17))

        img1 = imageio.imread(f"{baseline_path}/{baseline[num]}")
        img2 = imageio.imread(f"{results1_path}/{resut1[num]}")
        img3 = imageio.imread(f"{results2_path}/{results2[num]}")

        plt.subplot(1, 3, 1)
        plt.title("Test")
        plt.imshow(img1)

        plt.subplot(1, 3, 2)
        plt.title("baseline")
        plt.imshow(img2)

        plt.subplot(1, 3, 3)
        plt.title("Extra Iterations")
        plt.imshow(img3)

        plt.savefig(f"data/compare/compare_{num}.jpg")


if __name__ == "__main__":
    compare_resuts("data/test_data/real", "data/result_Zero_DCE++/baseline", "data/result_Zero_DCE++/extra_iterations")
