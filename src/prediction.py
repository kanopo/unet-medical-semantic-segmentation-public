import os
import PIL
import PIL.Image as Image
import torch
from model import IUNET
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from prediction_analysis.get_data_from_image import get_pixels_from_photo_labeled, get_values_distribution
import seaborn as sns
import matplotlib.pyplot as plt

def prediction_function(images_in_folder, input_folder, model):

    images = []
    masks = []
    for i in images_in_folder:
        image = PIL.Image.open(os.path.join(input_folder, i))
        image = image.convert("L")
        old_w, old_h = image.size
        
        new_w = 1280
        new_h = 876
        
        resized_image = Image.new(image.mode, (new_w, new_h), 0)
        resized_image.paste(image, ((new_w - old_w) // 2, (new_h - old_h) // 2))
        
        array = np.asarray(resized_image)
        array = array.astype(np.float32)
        array = array / 255
        
        tensor = TF.to_tensor(array)
        
        tensor = tensor.unsqueeze(0)
        
        prediction = model(tensor)
        
        prediction = prediction.squeeze(0)
        
        prediction = prediction.permute(1, 2, 0)
        
        prediction = prediction.squeeze(2)

        
        prediction = prediction.detach().numpy()
        
        prediction = (prediction > 0.2)
        
        predicted_image = Image.fromarray(prediction)

        images.append(resized_image)
        masks.append(predicted_image)
        # WARN: remove break for full folder
        break
    return images, masks

def extract_femur_from_prediction(images, masks):
    extracted_femurs = []
    # TODO: estrarre il femore dalla maschera predetta
    for i, (image, mask) in enumerate(zip(images, masks)):
        extracted_femur = get_pixels_from_photo_labeled(image, mask)
        # TODO: capire cosa fare delle immagini originali e di quelle estratte
        # extracted_femur.save(str(output_folder) + str(f"{i}_extracted_femur.png"))
        # mask.save(str(output_folder) + str(f"{i}_mask.png"))
        # image.save(str(output_folder) + str(f"{i}_image.png"))
        # extracted_femur.show()
        extracted_femurs.append(extracted_femur)

    return extracted_femurs

def extracted_data_distribution(extracted_femurs):
    values = []
    for i, (extracted_femur) in enumerate(extracted_femurs):
        value = get_values_distribution(image=extracted_femur, include_zero_val=False)
        data = pd.DataFrame(value, columns=["pixel_values"])
        # pd.DataFrame(value).to_csv(str(output_folder) + f"{i}_distribution_value.csv", index=False)
        values.append(data)

    return values

def graphs_plotting(data: list[pd.DataFrame]):
    for i, (distribution) in enumerate(data):
        # print(f"immagine {i} got mean {np.mean(distribution, axis=0)}")

        # graph = sns.displot(distribution, kde=True)
        # graph.savefig("./test.png")
        #
        # series: pd.Series = distribution.squeeze()
        #
        # graph = sns.displot(series, kde=True)
        # graph.savefig("./test_absolute.png")

        # distribution.columns = ["value"]
        # data_freq = distribution.groupby(["value"]).count()
        # data_freq.to_csv("./test.csv")

        # series: pd.Series = distribution.squeeze()
        # dataframe = series.value_counts().to_frame()
        # dataframe.columns = ["value", "freq"]
        # print(dataframe)

        # data_freq.columns= ["value", "freq"] 

        # plt.hist(x=data_freq["value"], weights=data_freq["freq"])
        # plt.show()

        # print(data_freq.iloc(axis=0)[:, 5])

        series = distribution.value_counts()
        dataframe = series.to_frame()
        print(dataframe)


        




def prediction(weights: str, input_folder: str, output_folder: str):

    model = torch.load(weights, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    percentile=[5, 10, 50, 90, 95]

    images_in_folder = os.listdir(input_folder)

    images, masks = prediction_function(images_in_folder, input_folder, model)

    extracted_femurs = extract_femur_from_prediction(images, masks)

    data = extracted_data_distribution(extracted_femurs)

    graphs_plotting(data)
  



if __name__ == "__main__":
    prediction(input_folder="../data/0.5 - nuovo ecografo/",output_folder="../nuovo_ecografo_results/", weights="../../../weights.pth")
