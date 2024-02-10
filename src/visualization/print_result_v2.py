import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from scores import intersection_over_union

matplotlib.use("Agg")


def print_function_v2(
    dataloader: torch.utils.data.DataLoader, model: nn.Module, device: torch.device.type
):

    images = []
    labels = []
    predictions = []
    accuracy = []

    accuracy_function = intersection_over_union()

    for x, y in dataloader:

        x, y = x.to(device), y.to(device)

        model = model.to(device)

        y_pred = model(x)

        acc = accuracy_function(y_pred, y)
        accuracy.append(acc.item())

        y_pred = (
            y_pred > 0.5
        ).float()  # converto la predizione in valori o 0 o 1, se sopra 0.5 => 1 se sotto 0.5 => 0

        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = y_pred.squeeze(0)

        x = x.cpu()
        y = y.cpu()
        y_pred = y_pred.cpu()

        x = x.permute(1, 2, 0)
        y = y.permute(1, 2, 0)
        y_pred = y_pred.permute(1, 2, 0)

        x = x.detach().numpy()
        y = y.detach().numpy()
        y_pred = y_pred.detach().numpy()

        images.append(x)
        labels.append(y)
        predictions.append(y_pred)

    for i in range(len(images)):
        fig = plt.figure(figsize=(19.2, 10.8))
        raws = 1
        cols = 3

        fig.add_subplot(raws, cols, 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        plt.title("Image " + str(i))

        fig.add_subplot(raws, cols, 2)
        plt.imshow(labels[i], cmap="gray")
        plt.axis("off")
        plt.title("Label " + str(i))

        fig.add_subplot(raws, cols, 3)
        plt.imshow(predictions[i], cmap="gray")
        plt.axis("off")
        plt.title(
            "Prediction "
            + str(i)
            + " with accuracy "
            + str(round(accuracy[i] * 100, 2))
            + "%"
        )

        plt.savefig(str(i) + ".png")
        plt.close()
