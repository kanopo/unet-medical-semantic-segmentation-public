import datetime
import logging
import time

import torch
from tqdm.auto import tqdm

from chat_logger import send_log


def cross_val_training_function(
    model: torch.nn.Module,
    device: torch.device,
    train_dataset: torch.utils.data.DataLoader,
    test_dataset: torch.utils.data.DataLoader,
    epochs: int,
    loss_function: torch.nn.Module,
    accuracy_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    path_weigths: str = "",
):
    scaler: torch.cuda.amp.GradScaler

    if str(device) == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    torch.autograd.set_detect_anomaly(False)

    best_loss = float("inf")
    best_accuracy = float(0)
    best_epoch = 0

    final_train_loss = []
    final_train_accuracy = []
    final_test_loss = []
    final_test_accuracy = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        test_loss = 0.0
        test_accuracy = 0.0

        mean_train_loss = 0.0
        mean_train_accuracy = 0.0
        mean_test_loss = 0.0
        mean_test_accuracy = 0.0

        """
        START TRAIN STEP
        """

        model.train()

        loop = tqdm(enumerate(train_dataset), total=len(train_dataset), leave=True)
        t0 = time.time()
        for batch, (x, y) in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            if str(device) == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                    y_pred = model(x)

                    loss = loss_function(y_pred, y)
                    accuracy = accuracy_function(y_pred, y)

                    train_loss += loss.item()
                    train_accuracy += accuracy.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(
                    train_loss=f"{round(train_loss / (batch + 1) * 100, 2)}%",
                    train_acc=f"{round(train_accuracy / (batch + 1) * 100, 2)}%",
                )

            else:
                y_pred = model(x)

                loss = loss_function(y_pred, y)
                accuracy = accuracy_function(y_pred, y)

                train_loss += loss.item()
                train_accuracy += accuracy.item()

                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(
                    train_loss=f"{round(train_loss / (batch + 1) * 100, 2)}%",
                    train_acc=f"{round(train_accuracy / (batch + 1) * 100, 2)}%",
                )

        mean_train_loss = train_loss / len(train_dataset)
        mean_train_accuracy = train_accuracy / len(train_dataset)

        scheduler.step(mean_train_loss)
        """
        END TRAIN STEP
        """

        # Comincia la parte di testing del modello
        model.eval()

        with torch.inference_mode():
            for x, y in test_dataset:
                x, y = x.to(device), y.to(device)

                y_pred = model(x)

                loss = loss_function(y_pred, y)
                accuracy = accuracy_function(y_pred, y)

                test_loss += loss.item()
                test_accuracy += accuracy.item()

        mean_test_loss = test_loss / len(test_dataset)
        mean_test_accuracy = test_accuracy / len(test_dataset)

        model.train()

        t1 = time.time() - t0

        logs = f"""
Epoch: {epoch}/{epochs}
Train loss: {round(mean_train_loss, 3)}
Train accuracy: {round(mean_train_accuracy * 100, 3)}%
Test loss: {round(mean_test_loss, 3)}
Test accuracy: {round(mean_test_accuracy * 100, 3)}%
Remaining time: {str(datetime.timedelta(seconds=((epochs - epoch) * t1))).split(".")[0]}
"""

        final_test_loss.append(mean_test_loss)
        final_test_accuracy.append(mean_test_accuracy)
        final_train_loss.append(mean_train_loss)
        final_train_accuracy.append(mean_train_accuracy)
        if epoch % 50 == 0:
            send_log(logs)

        logging.info(logs)

        # if model improve accuracy and loss, save the model with a reference to the epoch so i can read the full values from logs
        if mean_test_loss < best_loss and mean_test_accuracy > best_accuracy:
            logging.info("New weights saved")
            best_loss = mean_test_loss
            best_accuracy = mean_test_accuracy
            best_epoch = epoch
            if len(path_weigths) > 0:
                torch.save(model, path_weigths)

    send_log(
        f"""
The best test loss is {round(best_loss, 3)},
the best accuracy is {round(best_accuracy, 3)}
acived on epoch {best_epoch}.

             """
    )
    return (
        final_train_loss,
        final_train_accuracy,
        final_test_loss,
        final_test_accuracy,
        best_epoch,
    )


def single_training_function(
    model: torch.nn.Module,
    device: torch.device,
    train_dataset: torch.utils.data.DataLoader,
    epochs: int,
    loss_function: torch.nn.Module,
    accuracy_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    path_weigths: str = "",
):
    scaler: torch.cuda.amp.GradScaler

    if str(device) == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    torch.autograd.set_detect_anomaly(False)

    best_loss = float("inf")
    best_accuracy = float(0)
    best_epoch = 0

    final_train_loss = []
    final_train_accuracy = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        mean_train_loss = 0.0
        mean_train_accuracy = 0.0

        """
        START TRAIN STEP
        """

        model.train()

        loop = tqdm(enumerate(train_dataset), total=len(train_dataset), leave=True)
        t0 = time.time()
        for batch, (x, y) in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            if str(device) == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                    y_pred = model(x)

                    loss = loss_function(y_pred, y)
                    accuracy = accuracy_function(y_pred, y)

                    train_loss += loss.item()
                    train_accuracy += accuracy.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(
                    train_loss=f"{round(train_loss / (batch + 1) * 100, 2)}%",
                    train_acc=f"{round(train_accuracy / (batch + 1) * 100, 2)}%",
                )

            else:
                y_pred = model(x)

                loss = loss_function(y_pred, y)
                accuracy = accuracy_function(y_pred, y)

                train_loss += loss.item()
                train_accuracy += accuracy.item()

                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(
                    train_loss=f"{round(train_loss / (batch + 1) * 100, 2)}%",
                    train_acc=f"{round(train_accuracy / (batch + 1) * 100, 2)}%",
                )

        mean_train_loss = train_loss / len(train_dataset)
        mean_train_accuracy = train_accuracy / len(train_dataset)

        scheduler.step(mean_train_loss)
        """
        END TRAIN STEP
        """

        t1 = time.time() - t0

        logs = f"""
Epoch: {epoch}/{epochs}
Train loss: {round(mean_train_loss * 100, 3)}%
Train accuracy: {round(mean_train_accuracy * 100, 3)}%
Remaining time: {str(datetime.timedelta(seconds=((epochs - epoch) * t1))).split(".")[0]}
"""

        final_train_loss.append(mean_train_loss)
        final_train_accuracy.append(mean_train_accuracy)
        if epoch % 50 == 0:
            send_log(logs)

        logging.info(logs)

        # if model improve accuracy and loss, save the model with a reference to the epoch so i can read the full values from logs
        if mean_train_loss < best_loss and mean_train_accuracy > best_accuracy:
            logging.info("New weights saved")
            best_loss = mean_train_loss
            best_accuracy = mean_train_accuracy
            best_epoch = epoch
            torch.save(model, path_weigths)

    send_log(
        f"""
The best test loss is {round(best_loss, 3)},
the best accuracy is {round(best_accuracy, 3)}
acived on epoch {best_epoch}.

             """
    )
    return (
        final_train_loss,
        final_train_accuracy,
        best_epoch,
    )
