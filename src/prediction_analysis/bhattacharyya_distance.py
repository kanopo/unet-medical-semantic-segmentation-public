import numpy as np
import math
import PIL.Image as Image
from get_histogram_from_image import get_values_distribution


def bhattacharyya_coefficient(distribution_1, distribution_2):
    return np.sum(np.sqrt(distribution_1 * distribution_2))


def bhattacharyya_distance(distribution1, distribution2):
    return -np.log(bhattacharyya_coefficient(distribution1, distribution2))


if __name__ == "__main__":
    image = Image.open("./../../data/0.3 - labels/155_json/img.png").convert("L")
    label = Image.open("./../../data/0.3 - labels/155_json/label.png").convert("L")

    # TODO: add the predicted mask and do the bhattacharyya_distance

    values = get_values_distribution(image, label)

    bd = bhattacharyya_distance(values, [1, 2, 3])
    print(bd)


