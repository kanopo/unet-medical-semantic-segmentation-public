
import pandas as pd

def save_dataset_metrics():
    data = pd.read_csv("../results/results.csv")
    dataset_metrics: pd.DataFrame = pd.DataFrame(columns=["5_percetile", "10_percentile","50_percentile","90_percentile", "95_percentile"])

    handmade_data = data[data["image_path"].str.contains("handmade")]
    model_data = data[data["image_path"].str.contains("model")]

    percentile_05_handmade = handmade_data["mean"].quantile(0.05)
    percentile_10_handmade = handmade_data["mean"].quantile(0.10)
    percentile_50_handmade = handmade_data["mean"].quantile(0.5)
    percentile_90_handmade = handmade_data["mean"].quantile(0.9)
    percentile_95_handmade = handmade_data["mean"].quantile(0.95)

    percentile_05_model = model_data["mean"].quantile(0.05)
    percentile_10_model = model_data["mean"].quantile(0.10)
    percentile_50_model = model_data["mean"].quantile(0.5)
    percentile_90_model = model_data["mean"].quantile(0.9)
    percentile_95_model = model_data["mean"].quantile(0.95)

    handmade_results = f"""
percentile_05_handmade: {percentile_05_handmade},
percentile_10_handmade: {percentile_10_handmade},
percentile_50_handmade: {percentile_50_handmade},
percentile_90_handmade: {percentile_90_handmade},
percentile_95_handmade: {percentile_95_handmade},
    """
    model_results = f"""
percentile_05_model: {percentile_05_model},
percentile_10_model: {percentile_10_model},
percentile_50_model: {percentile_50_model},
percentile_90_model: {percentile_90_model},
percentile_95_model: {percentile_95_model},
    """

    
    dataset_metrics.loc[len(dataset_metrics)] = [
            percentile_05_handmade,
            percentile_10_handmade,
            percentile_50_handmade,
            percentile_90_handmade,
            percentile_95_handmade,
                ]
    dataset_metrics.loc[len(dataset_metrics)] = [
            percentile_05_model,
            percentile_10_model,
            percentile_50_model,
            percentile_90_model,
            percentile_95_model,
                ]

    
    dataset_metrics.to_csv("../results/dataset_results.csv", index=False)
    data.to_excel("../results/results.xlsx", index=False)
    dataset_metrics.to_excel("../results/dataset_results.xlsx", index=False)


    data = pd.read_csv("../results/results.csv", names=["image_path", "mean", "variance", "standard_deviation", "5_percetile", "10_percentile","50_percentile","90_percentile", "95_percentile"])

    handmade_data = data[data["image_path"].str.contains("handmade")]
    model_data = data[data["image_path"].str.contains("model")]

    handmade_data.to_csv("../results/handmade_data.csv", index=False)
    model_data.to_csv("../results/model_data.csv", index=False)

    handmade_data.to_excel("../results/handmade_data.xlsx", index=False, header=True)
    model_data.to_excel("../results/model_data.xlsx", index=False, header=True)


if __name__ == "__main__":
    save_dataset_metrics()

