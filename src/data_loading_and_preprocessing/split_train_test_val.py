
import random
import numpy as np

def split_train_test_val(
        images,
        labels,
        train_size: float = 0.70,
        test_size: float = 0.25,
):
    """
    Function used to split the dataset in three parts, the 95% of the dataset will be composed of the training dataset
    and the test dataset, I chose this proportion because I use the validation set only for data visualization referred
     to the accuracy that the model can achieve.
    :param images: array of all the images in teh dataset
    :param labels: array of all the labels in the dataset
    :param train_size: this size is expressed in percentage, and represent the train dataset size
    :param test_size: this size is expressed in percentage, and represent the test dataset size
    :param val_size: this size is expressed in percentage, and represent the val dataset size
    :return: 6 arrays a couple of arrays(image and labels) for train, test and val set
    """

    # array with all the indexes for the images and labels
    indexes: list[any] = np.arange(0, len(images))

    random.shuffle(indexes)

    # train array goes from the 0 element to the split1 element
    split1 = int(train_size * len(images))

    # test array goes from split1 + 1 to split2
    split2 = int((train_size + test_size) * len(images))

    train_indexes = indexes[:split1]
    test_indexes = indexes[split1:split2]
    val_indexes = indexes[split2:]

    train_images: list[any] = []
    train_labels: list[any] = []

    test_images: list[any] = []
    test_labels: list[any] = []

    val_images: list[any] = []
    val_labels: list[any] = []

    for index in train_indexes:
        train_images.append(images[index])
        train_labels.append(labels[index])

    for index in test_indexes:
        test_images.append(images[index])
        test_labels.append(labels[index])

    for index in val_indexes:
        val_images.append(images[index])
        val_labels.append(labels[index])

    return train_images, train_labels, test_images, test_labels, val_images, val_labels