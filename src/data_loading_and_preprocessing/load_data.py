import os
from PIL import Image


def load_data(path_to_data: str):

    items_in_folder: list[str] = []

    try:
        items_in_folder = os.listdir(path_to_data)
    except IOError:
        print("IOException during data loading")

    labeled_folders: list[str] = []

    for items in items_in_folder:
        if items.endswith("_json"):
            labeled_folders.append(items)

    images: list[any] = []
    labels: list[any] = []

    for labeled in labeled_folders:
        # open image and label
        image = Image.open(os.path.join(path_to_data, labeled + "/img.png"))
        label = Image.open(os.path.join(path_to_data, labeled + "/label.png"))

        # convert image and label in grayscale
        image = image.convert("L")
        label = label.convert("L")

        # use a threshold to make label 0 o 1 values
        label = label.point(lambda p: 255 if p > 32 else 0)

        images.append(image)
        labels.append(label)

    return images, labels
