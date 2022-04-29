import matplotlib.pyplot as plt
import imageio
import os


plt.rcParams.update({"figure.max_open_warning": 0})


original = os.listdir("data/test_data/real")
with_attention = os.listdir("data/result_Zero_DCE++/with_cbam/real")
without_attention = os.listdir("data/result_Zero_DCE++/without_cbam/real")

NUM = len(original)


for num in range(NUM):
    plt.figure(figsize=(20, 17))

    img1 = imageio.imread(f"data/test_data/real/{original[num]}")
    img2 = imageio.imread(f"data/result_Zero_DCE++/without_cbam/real/{without_attention[num]}")
    img3 = imageio.imread(f"data/result_Zero_DCE++/with_cbam/real/{with_attention[num]}")

    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(img1)

    plt.subplot(1, 3, 2)
    plt.title("without_attention")
    plt.imshow(img2)

    plt.subplot(1, 3, 3)
    plt.title("with_attention")
    plt.imshow(img3)

    plt.savefig(f"data/compare/compare_{num}.jpg")
