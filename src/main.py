import tomli
import argparse
import torch

from cross_validation import cross_validation
from cross_validation_normal import cross_validation as cross_validation_normal
from chat_logger import send_log
from single_train import training
from prediction import prediction
import logging


def get_args():
    parser = argparse.ArgumentParser(
        prog="UNET cli for semantic segmentation",
        description="cli application used to interact with neural network",
    )

    # Training related args
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        choices=["train", "cross_validation", "predict"],
        help="Chose the mode to run the model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    with open("./config.toml") as config_file:
        configs = tomli.loads(config_file.read())

    MODE = get_args().mode

    GENERAL = configs["GENERAL"]
    TRAINING = configs["TRAINING"]
    CROSS_VALIDATION = configs["CROSS_VALIDATION"]
    PREDICTION = configs["PREDICTION"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match (MODE):
        case "train":
            starting_train_logs = f"""
Starting new training session with parameters:
    Epochs: {TRAINING["EPOCHS"]},
    Image width: {GENERAL["IMAGE_WIDTH"]}px,
    Image height: {GENERAL["IMAGE_HEIGHT"]}px,
    Learning rate: {TRAINING["LEARNING_RATE"]},
    Weight decay: {TRAINING["WEIGHT_DECAY"]},
    Batch size: {TRAINING["BATCH_SIZE"]},
    Convolution dropout: {TRAINING["CONV_DROPOUT"]},
    Final dropout: {TRAINING["FINAL_DROPOUT"]},
    Device: {DEVICE},
    Saving results: {TRAINING["SAVE_WEIGHTS"]},
    Path final weights: {TRAINING["PATH_WEIGHTS"]}
                  """
            print(starting_train_logs)
            send_log(starting_train_logs)

            training(
                EPOCHS=TRAINING["EPOCHS"],
                PATH_TO_DATA=TRAINING["PATH_TRAINING_DATA"],
                IMAGE_W=GENERAL["IMAGE_WIDTH"],
                IMAGE_H=GENERAL["IMAGE_HEIGHT"],
                LEARNING_RATE=TRAINING["LEARNING_RATE"],
                WEIGHT_DECAY=TRAINING["WEIGHT_DECAY"],
                BATCH_SIZE=TRAINING["BATCH_SIZE"],
                DEVICE=DEVICE,
                SAVE_RESULTS=TRAINING["SAVE_WEIGHTS"],
                CONV_DROPOUT=TRAINING["CONV_DROPOUT"],
                FINAL_DROPOUT=TRAINING["FINAL_DROPOUT"],
                PATH_WEIGHTS=TRAINING["PATH_WEIGHTS"],
            )

        case "cross_validation":
            starting_cross_validation_logs = f"""
Starting new cross validation session with parameters:
    Epochs: {CROSS_VALIDATION["EPOCHS"]},
    Folds: {CROSS_VALIDATION["NUM_FOLDS"]}
    Image width: {GENERAL["IMAGE_WIDTH"]}px,
    Image height: {GENERAL["IMAGE_HEIGHT"]}px,
    Learning rate: {CROSS_VALIDATION["LEARNING_RATE"]},
    Weight decay: {CROSS_VALIDATION["WEIGHT_DECAY"]},
    Batch size: {CROSS_VALIDATION["BATCH_SIZE"]},
    Convolution dropout: {CROSS_VALIDATION["CONV_DROPOUT"]},
    Final dropout: {CROSS_VALIDATION["FINAL_DROPOUT"]},
    Device: {DEVICE},
                  """
            print(starting_cross_validation_logs)
            send_log(starting_cross_validation_logs)

            cross_validation(
                KFOLD=CROSS_VALIDATION["NUM_FOLDS"],
                EPOCHS=CROSS_VALIDATION["EPOCHS"],
                PATH_TO_DATA=CROSS_VALIDATION["PATH_CROSS_VALIDATION_DATA"],
                IMAGE_W=GENERAL["IMAGE_WIDTH"],
                IMAGE_H=GENERAL["IMAGE_HEIGHT"],
                LEARNING_RATE=CROSS_VALIDATION["LEARNING_RATE"],
                WEIGHT_DECAY=CROSS_VALIDATION["WEIGHT_DECAY"],
                BATCH_SIZE=CROSS_VALIDATION["BATCH_SIZE"],
                DEVICE=DEVICE,
                CONV_DROPOUT=CROSS_VALIDATION["CONV_DROPOUT"],
                FINAL_DROPOUT=CROSS_VALIDATION["FINAL_DROPOUT"],
            )

        case "predict":
            starting_predictions_logs = f"""
Starting predictions with params:

            """
            prediction(
                options=PREDICTION["OPTIONS"],
                metrics=PREDICTION["METRICS"],
                percentile=PREDICTION["PERCENTILE"],
                weights=PREDICTION["PATH_WEIGHTS"],
                device=DEVICE,
                input_folder=PREDICTION["INPUT_FOLDER"],
                output_folder=PREDICTION["OUTPUT_FOLDER"],
            )

        case "default":
            # cross validation without data augmentation and without modified unet
            starting_cross_validation_logs = f"""
                Starting new cross validation session with parameters:
                Epochs: {CROSS_VALIDATION["EPOCHS"]},
                Folds: {CROSS_VALIDATION["NUM_FOLDS"]}
                Image width: {GENERAL["IMAGE_WIDTH"]}px,
                Image height: {GENERAL["IMAGE_HEIGHT"]}px,
                Learning rate: {CROSS_VALIDATION["LEARNING_RATE"]},
                Weight decay: {CROSS_VALIDATION["WEIGHT_DECAY"]},
                Batch size: {CROSS_VALIDATION["BATCH_SIZE"]},
                Convolution dropout: {CROSS_VALIDATION["CONV_DROPOUT"]},
                Final dropout: {CROSS_VALIDATION["FINAL_DROPOUT"]},
                Device: {DEVICE},
                              """

            logging.basicConfig(
                filename="normale.log",
                level=logging.DEBUG,
                format="%(asctime)s:%(levelname)s:%(message)s",
            )

            cross_validation_normal(
                KFOLD=CROSS_VALIDATION["NUM_FOLDS"],
                EPOCHS=CROSS_VALIDATION["EPOCHS"],
                PATH_TO_DATA=CROSS_VALIDATION["PATH_CROSS_VALIDATION_DATA"],
                IMAGE_W=GENERAL["IMAGE_WIDTH"],
                IMAGE_H=GENERAL["IMAGE_HEIGHT"],
                LEARNING_RATE=CROSS_VALIDATION["LEARNING_RATE"],
                WEIGHT_DECAY=CROSS_VALIDATION["WEIGHT_DECAY"],
                BATCH_SIZE=CROSS_VALIDATION["BATCH_SIZE"],
                DEVICE=DEVICE,
                CONV_DROPOUT=CROSS_VALIDATION["CONV_DROPOUT"],
                FINAL_DROPOUT=CROSS_VALIDATION["FINAL_DROPOUT"],
            )

        case _:
            # WARN: main started without mode setted
            print("No chars")
