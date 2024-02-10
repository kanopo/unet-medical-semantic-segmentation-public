# TODO:
#   - prestare attenzione agli errori di out of memory
from scores import intersection_over_union, dice_bce_loss
from data_loading_and_preprocessing import (
    femur_dataset,
    load_data,
    augmentation,
)

from model import IUNET

from torch.utils.data import DataLoader

from training import single_training_function


import albumentations as A
import cv2

import torch

from visualization import (
    plot_loss_over_epochs_train,
    plot_accuracy_over_epochs_train,
)

import os
import numpy as np

from chat_logger import send_log, send_pred


def training(
    EPOCHS: int = 300,
    PATH_TO_DATA: str = "./../data/0.3 - labels/",
    IMAGE_W: int = 1280,
    IMAGE_H: int = 876,
    LEARNING_RATE: float = 1e-3,
    WEIGHT_DECAY: float = 1e-8,
    BATCH_SIZE: int = 4,
    DEVICE: torch.device = torch.device("cpu"),
    SAVE_RESULTS: bool = False,
    CONV_DROPOUT: float = 0.0,
    FINAL_DROPOUT: float = 0.0,
    PATH_WEIGHTS: str = "./weigths.pth"
):
    loss_function = dice_bce_loss()
    accuracy_function = intersection_over_union()

    loss_function = loss_function.to(DEVICE)
    accuracy_function = accuracy_function.to(DEVICE)

    """
    Carico le immagini contenute nella cartella dentro alle variabili images e labels
    """
    images, labels = load_data(path_to_data=PATH_TO_DATA)

    train_augment = A.Compose(
        [
            A.CenterCrop(width=IMAGE_W, height=IMAGE_H),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=35, p=0.5, border_mode=cv2.BORDER_CONSTANT
            ),  # il BORDER_CONSTANT è li perchè senno albumentations specchia i bordi, così lascia una parte nera.
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
        ]
    )

    # ora che ho separato le mie immagini di validation riunisco le immagini di test con quelle di train

    train_images, train_labels = augmentation.data_augmentation(
        images=images,
        labels=labels,
        data_augmentation=train_augment,
    )

    train = femur_dataset.FemurDataset(
        train_images,
        train_labels,
    )

    train_dataset = DataLoader(
        dataset=train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=torch.cuda.is_available(),
    )

    model = IUNET(conv_dropout=CONV_DROPOUT, final_dropout=FINAL_DROPOUT).to(
        device=DEVICE
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, min_lr=1e-6, cooldown=0, verbose=True
    )

    (
        train_loss,
        train_accuracy,
        best_epoch,
    ) = single_training_function(
        model=model,
        device=DEVICE,
        loss_function=loss_function,
        accuracy_function=accuracy_function,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        train_dataset=train_dataset,
        path_weigths=PATH_WEIGHTS,
    )

    print("Best loss " + str(train_loss[best_epoch]))
    print("Best accuracy " + str(train_accuracy[best_epoch]))

    plot_loss_over_epochs_train(train_loss, str("loss.png"))
    plot_accuracy_over_epochs_train(train_accuracy, str("accuracy.png"))

    send_pred(str("./loss.png"))
    send_pred(str("./accuracy.png"))

    if SAVE_RESULTS:
        # move weigth to the result folder
        # TODO: move weights, loss and accuracy images in the correct folders

        pass
    else:
        # delete weights and images about loss and accuracy after the sand
        os.remove("./loss.png")
        os.remove("./accuracy.png")
        os.remove(PATH_WEIGHTS)


if __name__ == "__main__":
    training()
