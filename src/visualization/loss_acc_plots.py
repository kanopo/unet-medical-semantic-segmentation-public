import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_accuracy_over_epochs_train_test(
    train: list[float], test: list[float], name: str
):
    plt.title("Accuracy over epochs")
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train)
    plt.plot(epochs, test)
    plt.legend(["Training Accuracy", "Test Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(name)
    plt.close()


def plot_loss_over_epochs_train_test(train: list[float], test: list[float], name: str):
    plt.title("Loss over epochs")
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train)
    plt.plot(epochs, test)
    plt.legend(["Training Loss", "Test Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(name)
    plt.close()


def plot_accuracy_over_epochs_train(train: list[float], name: str):
    plt.title("Accuracy over epochs")
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train)
    plt.legend(["Training Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(name)
    plt.close()


def plot_loss_over_epochs_train(train: list[float], name: str):
    plt.title("Loss over epochs")
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train)
    plt.legend(["Training Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(name)
    plt.close()
