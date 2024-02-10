# TODO::
#   - prestare attenzione agli errori di out of memory
import logging
from scores import intersection_over_union, dice_bce_loss
from data_loading_and_preprocessing import (
    femur_dataset,
    load_data,
    augmentation,
)


from model import OUNET

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from training import cross_val_training_function

import albumentations as A
import cv2

import torch
import PIL.Image as Image

from visualization import (
    print_function_v2,
    get_prediction,
    plot_loss_over_epochs_train_test,
    plot_accuracy_over_epochs_train_test,
)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from prediction_analysis.get_data_from_image import (
    get_pixels_from_photo_labeled,
    get_values_distribution,
)

from metrics import save_dataset_metrics


class WrapperDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image, label = augmentation.single_data_augmentation(
            image=image, label=label, data_augmentation=self.transform
        )
        return image, label

    def __len__(self):
        return len(self.dataset)


def cross_validation(
    KFOLD: int = 5,
    EPOCHS: int = 300,
    PATH_TO_DATA: str = "./../data/0.3 - labels/",
    IMAGE_W: int = 1280,
    IMAGE_H: int = 876,
    LEARNING_RATE: float = 1e-3,
    WEIGHT_DECAY: float = 1e-8,
    BATCH_SIZE: int = 4,
    DEVICE: torch.device = torch.device("cpu"),
    CONV_DROPOUT: float = 0.0,
    FINAL_DROPOUT: float = 0.0,
):
    best_loss_folds: list[float] = []
    best_accuracy_folds: list[float] = []

    loss_function = dice_bce_loss()
    accuracy_function = intersection_over_union()

    loss_function = loss_function.to(DEVICE)
    accuracy_function = accuracy_function.to(DEVICE)

    images, labels = load_data(path_to_data=PATH_TO_DATA)

    dataset = femur_dataset.FemurDataset(images, labels)
    kfold = KFold(n_splits=KFOLD, shuffle=False)

    train_augment = A.Compose(
        [
            A.CenterCrop(width=IMAGE_W, height=IMAGE_H),
        ]
    )

    test_augment = A.Compose(
        [
            A.CenterCrop(width=IMAGE_W, height=IMAGE_H),
        ]
    )

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(train_ids)
        # print(test_ids)
        #         send_log(
        #             f"""
        # Train ids: {train_ids},
        # Test ids: {test_ids}
        #                  """
        #         )

        train_dataset = DataLoader(
            dataset=WrapperDataset(dataset=dataset, transform=train_augment),
            batch_size=BATCH_SIZE,
            num_workers=os.cpu_count() // 2,
            pin_memory=torch.cuda.is_available(),
            sampler=SubsetRandomSampler(train_ids),
        )

        test_dataset = DataLoader(
            dataset=WrapperDataset(dataset=dataset, transform=test_augment),
            batch_size=1,
            num_workers=os.cpu_count() // 2,
            pin_memory=torch.cuda.is_available(),
            sampler=SubsetRandomSampler(test_ids),
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
            test_loss,
            test_accuracy,
            best_epoch,
        ) = cross_val_training_function(
            model=model,
            device=DEVICE,
            loss_function=loss_function,
            accuracy_function=accuracy_function,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=EPOCHS,
            test_dataset=test_dataset,
            train_dataset=train_dataset,
            path_weigths=str("fold_" + str(fold) + "_weights.pth"),
        )

        print(f"Results fold number {fold}\n\n")
        print(f"Best loss {str(test_loss[best_epoch])}")
        best_loss_folds.append(test_loss[best_epoch])

        print(f"Best accuracy {str(test_accuracy[best_epoch])}")
        best_accuracy_folds.append(test_accuracy[best_epoch])

        plot_loss_over_epochs_train_test(
            train_loss, test_loss, str("fold_" + str(fold) + "_loss.png")
        )
        plot_accuracy_over_epochs_train_test(
            train_accuracy, test_accuracy, str("fold_" + str(fold) + "_accuracy.png")
        )

        logging.info(f"Fold {fold} completed")
        logging.info(f"Best loss {str(test_loss[best_epoch])}")
        logging.info(f"Best accuracy {str(test_accuracy[best_epoch])}")
        logging.info(f"Accuracy saved in fold_{fold}_accuracy.png")
        logging.info(f"Loss saved in fold_{fold}_loss.png")

        # send_pred(str("./fold_" + str(fold) + "_loss.png"))
        # send_pred(str("./fold_" + str(fold) + "_accuracy.png"))

        # WARN: the model use gpu to predict final image(This is really big, so if the model fail try cpu)

        model = torch.load(f"./fold_{fold}_weights.pth")

        test_images: list[Image.Image] = []
        test_labels: list[Image.Image] = []
        for i in test_ids:
            image, mask = dataset.__getitem__(i)
            test_images.append(image)
            test_labels.append(mask)

        get_prediction(
            f"./fold_{fold}_weights.pth",
            names=test_ids,
            images=test_images,
            labels=test_labels,
            device=DEVICE,
            path="../results/",
        )

        os.remove(f"./fold_{fold}_weights.pth")
        # os.remove(f"./fold_{fold}_loss.png")
        # os.remove(f"./fold_{fold}_accuracy.png")

    print(f"Mean loss across fold {np.mean(best_loss_folds)}")
    print(f"Mean accuracy across fold {np.mean(best_accuracy_folds)}")

    logging.info(f"Mean loss across fold {np.mean(best_loss_folds)}")
    logging.info(f"Mean accuracy across fold {np.mean(best_accuracy_folds)}")
    #
    #
    # TODO:
    #   - analizzare le immagini con le maschere e salvare i risultati in csv che puntatno alle immagini

    result_dirs = os.listdir("../results")
    data: pd.DataFrame = pd.DataFrame(
        columns=[
            "image_path",
            "mean",
            "variance",
            "standard_deviation",
            "5_percetile",
            "10_percentile",
            "50_percentile",
            "90_percentile",
            "95_percentile",
        ]
    )

    for dir in result_dirs:
        image = Image.open(os.path.join("../results", dir, "image.png"))
        mask = Image.open(os.path.join("../results", dir, "mask.png"))
        prediction = Image.open(os.path.join("../results", dir, "prediction.png"))

        extracted_femur_handmade = get_pixels_from_photo_labeled(image, mask)
        extracted_femur_model = get_pixels_from_photo_labeled(image, prediction)

        extracted_femur_handmade.save(
            os.path.join("../results", dir, "extracted_femur_handmade.png")
        )
        extracted_femur_model.save(
            os.path.join("../results", dir, "extracted_femur_model.png")
        )

        distribution_handmade = get_values_distribution(extracted_femur_handmade)
        distribution_model = get_values_distribution(extracted_femur_model)

        # salvo le distribuzioni cosi da non dover allenare il modello ogni volta che voglio dei dati
        pd.DataFrame(distribution_handmade).to_csv(
            f"../results/{dir}/handmade_data.csv", index=False
        )
        pd.DataFrame(distribution_model).to_csv(
            f"../results/{dir}/model_data.csv", index=False
        )

        mean_handmade = np.mean(distribution_handmade)
        mean_model = np.mean(distribution_model)

        variance_handmade = np.var(distribution_handmade)
        variance_model = np.var(distribution_model)

        std_handmade = np.std(distribution_handmade)
        std_model = np.std(distribution_model)

        perc_handmade = np.percentile(distribution_handmade, [5, 10, 50, 90, 95])
        perc_model = np.percentile(distribution_model, [5, 10, 50, 90, 95])

        data.loc[len(data)] = [
            f"./{dir}/extracted_femur_handmade.png",
            mean_handmade,
            variance_handmade,
            std_handmade,
            perc_handmade[0],
            perc_handmade[1],
            perc_handmade[2],
            perc_handmade[3],
            perc_handmade[4],
        ]

        data.loc[len(data)] = [
            f"./{dir}/extracted_femur_model.png",
            mean_model,
            variance_model,
            std_model,
            perc_model[0],
            perc_model[1],
            perc_model[2],
            perc_model[3],
            perc_model[4],
        ]

    data.to_csv("../results/results.csv", index=False)
    save_dataset_metrics()


if __name__ == "__main__":
    cross_validation(KFOLD=5, EPOCHS=5, IMAGE_W=128, IMAGE_H=128)
