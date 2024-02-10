import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_hist(unique, counts, path, title):

    plt.hist(x=unique, weights=counts, bins=np.arange(min(unique), max(unique) + 1, 1))
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel count")
    plt.savefig(path)
    plt.close()

def print_hist_scaled(unique, counts, path, title):

    counts = counts / counts.max()

    plt.hist(x=unique, weights=counts, bins=np.arange(min(unique), max(unique) + 1, 1))
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel count (scaled)")
    plt.savefig(path)
    plt.close()

def print_hist_2_dist(unique1, unique2, counts1, counts2, path, title):

    counts2 = counts2 * (-1)

    colors = ["lime", "tan"]
    plt.hist(x=unique1, weights=counts1, bins=np.arange(min(unique1), max(unique1) + 1, 1), alpha=0.5, label='handmade', color="red")
    plt.hist(x=unique2, weights=counts2, bins=np.arange(min(unique2), max(unique2) + 1, 1), alpha=0.5, label='model', color="blue")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel count")
    plt.savefig(path)
    plt.close()


def print_hist_2_dist_scaled(unique1, unique2, counts1, counts2, path, title):

    counts1 = counts1 / counts1.max()
    counts2 = counts2 / counts2.max()
    counts2 = counts2 * (-1)

    colors = ["lime", "tan"]
    plt.hist(x=unique1, weights=counts1, bins=np.arange(min(unique1), max(unique1) + 1, 1), alpha=0.5, label='handmade', color="red")
    plt.hist(x=unique2, weights=counts2, bins=np.arange(min(unique2), max(unique2) + 1, 1), alpha=0.5, label='model', color="blue")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel count (scaled)")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    path = "../results/"
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)):
            path_dir = os.path.join(path, i)

            handmade: pd.DataFrame = pd.read_csv(os.path.join(path_dir, "handmade_data.csv"))
            model: pd.DataFrame = pd.read_csv(os.path.join(path_dir, "model_data.csv"))


            array_handmade: np.array = handmade.values
            array_model: np.array = model.values


            handmade_mean = array_handmade.mean()
            handmade_deviation = array_handmade.std()

            model_mean = array_model.mean()
            model_deviation = array_model.std()

            dataframe = pd.DataFrame(columns=["handmade_mean", "handmade_deviation", "model_mean", "model_deviation"])
            dataframe.loc[len(dataframe)] = [
                    handmade_mean,
                    handmade_deviation,
                    model_mean,
                    model_deviation,
                    ]
            dataframe.to_csv(f"{path_dir}/mean_deviation.csv", index=False)
            dataframe.to_excel(f"{path_dir}/mean_deviation.xlsx", index=False, header=True)


            # la variabile unique contiene tutti i valori unici del data set
            # la variabile counts contiene nella stesas posizione del suo valore unico(in unque) il numero di volte che il valore compare

            unique_handmade, counts_handmade = np.unique(array_handmade, return_counts=True)
            unique_model, counts_model = np.unique(array_model, return_counts=True)

            print_hist(unique_handmade, counts_handmade, f"{path_dir}/handmade_non_scaled_hist.png", title="Handmade")
            print_hist_scaled(unique_handmade, counts_handmade, f"{path_dir}/handmade_scaled_hist.png", title="Handmade")

            print_hist(unique_model, counts_model, f"{path_dir}/model_non_scaled_hist.png", title="Model")
            print_hist_scaled(unique_model, counts_model, f"{path_dir}/model_scaled_hist.png", title="Model")

            print_hist_2_dist(unique_handmade, unique_model, counts_handmade, counts_model, path=f"{path_dir}/handmade_vs_model_non_scaled.png", title="Handmade Vs Model")
            print_hist_2_dist_scaled(unique_handmade, unique_model, counts_handmade, counts_model, path=f"{path_dir}/handmade_vs_model_scaled.png", title="Handmade Vs Model")



    
