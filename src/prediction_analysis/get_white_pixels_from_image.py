import PIL.Image as Image
import numpy as np


def get_pixels_from_photo_labeled(image: Image.Image, label: Image.Image):
    image_array = np.asarray(image)
    label_array = np.asarray(label)

    label_array = label_array > 0
    extracted_mask_array = image_array * label_array

    extracted_mask = Image.fromarray(extracted_mask_array).convert("L")

    return extracted_mask


def non_zero_array_extraction(image: Image) -> np.array:
    base_array = np.asarray(image)
    flat = base_array.flatten()

    non_zero = flat[np.where(flat != 0)]
    return non_zero


def mean(array: np.array) -> np.float32:
    return np.mean(array)


def std(array: np.array) -> np.float32:
    return np.std(array)


if __name__ == "__main__":
    # convertito le immagini in grayscale
    # image = Image.open("./../../data/0.3 - labels/155_json/img.png").convert("L")
    # label = Image.open("./../../data/0.3 - labels/155_json/label.png").convert("L")
    image = Image.open(
        "./../test_prediction/original/ASLERPR000370243_CIRILA_VITO_19870128_20230223101823715-1.jpg.png"
    ).convert("L")
    label = Image.open(
        "./../test_prediction/prediction/ASLERPR000370243_CIRILA_VITO_19870128_20230223101823715-1.jpg.png"
    ).convert("L")

    # image.show()
    # label.show()

    extracted_mask = get_pixels_from_photo_labeled(image, label)

    non_zero = non_zero_array_extraction(extracted_mask)
    # extracted_mask.show()
    mean = mean(non_zero)
    std = std(non_zero)

    print("Mean: " + str(mean))
    print("Standard deviation: " + str(std))
