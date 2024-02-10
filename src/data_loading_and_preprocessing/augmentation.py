import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
import PIL
import matplotlib.pyplot as plt


def single_data_augmentation(image, label, data_augmentation=None):

    image_array = np.asarray(image)
    label_array = np.asarray(label)

    transformed = data_augmentation(image=image_array, mask=label_array)

    transformed_image_array = transformed["image"]
    transformed_label_array = transformed["mask"]

    norm_image_array = transformed_image_array.astype(np.float32)
    norm_label_array = transformed_label_array.astype(np.float32)

    norm_image_array = norm_image_array / 255
    norm_label_array = norm_label_array / 255

    tensor_image = TF.to_tensor(norm_image_array)
    tensor_label = TF.to_tensor(norm_label_array)

    return tensor_image, tensor_label


def data_augmentation(images, labels, data_augmentation=None):

    final_images = []
    final_labels = []

    for i in range(len(images)):
        image_array = np.asarray(images[i])
        label_array = np.asarray(labels[i])

        transformed = data_augmentation(image=image_array, mask=label_array)

        transformed_image_array = transformed["image"]
        transformed_label_array = transformed["mask"]

        norm_image_array = transformed_image_array.astype(np.float32)
        norm_label_array = transformed_label_array.astype(np.float32)

        norm_image_array = norm_image_array / 255
        norm_label_array = norm_label_array / 255

        tensor_image = TF.to_tensor(norm_image_array)
        tensor_label = TF.to_tensor(norm_label_array)

        final_images.append(tensor_image)
        final_labels.append(tensor_label)

    return final_images, final_labels


if __name__ == "__main__":
    image = PIL.Image.open("./../../data/0.3 - labels/0_json/img.png")
    image = image.convert("L")

    tensor = TF.to_tensor(image)

    mean = tensor.mean([1, 2])
    std = tensor.std([1, 2])

    print(mean)
    print(std)

    tensor = tensor / 255

    mean = tensor.mean([1, 2])
    std = tensor.std([1, 2])

    print(mean)
    print(std)
