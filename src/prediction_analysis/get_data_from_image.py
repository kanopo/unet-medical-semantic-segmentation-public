import PIL.Image as Image
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from .get_white_pixels_from_image import get_pixels_from_photo_labeled

# matplotlib.use("QtAgg")


def get_values_distribution(image: Image.Image, include_zero_val: bool = False):
    # convert image in numpy array
    array = np.asarray(image)
    # value is an array of all the values present in the image [0, 255]
    # count is the number of istances founded in the image about the value in the same index
    if include_zero_val:
        values, count = np.unique(array, return_counts=True)
    else:
        values, count = np.unique(array[np.nonzero(array)], return_counts=True)

    return np.repeat(values, count)


if __name__ == "__main__":
    image = Image.open(
        "./../test_prediction/original/test.png"
    ).convert("L")
    label = Image.open(
        "./../test_prediction/prediction/test.png"
    ).convert("L")

    extraced_image = get_pixels_from_photo_labeled(image, label)

    extraced_image.save("./extracted_femur.png")

    values = get_values_distribution(image=extraced_image, include_zero_val=False)

    mean = np.mean(values)
    print(f"\nMedia: {mean}")
    variance = np.var(values)
    print(f"Varianza: {variance}")
    std = np.std(values)
    print(f"Deviazione standard: {std}")
    perc = np.percentile(values, [5, 50, 95])
    print(f"5 percentile: {perc[0]}")
    print(f"50 percentile: {perc[1]}")
    print(f"95 percentile: {perc[2]}\n")

    plt.hist(
        x=values,
        bins=63,
    )
    plt.savefig("./image.png")
