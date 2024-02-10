import PIL.Image as Image
import numpy as np
import torchvision.transforms.functional as TF
import torch
import os

def get_prediction(weigths: str, names: list[np.integer], images: list[Image.Image], labels: list[Image.Image], device, path: str):
    # model = torch.load(weigths, map_location=device)

    model = torch.load(weigths).to(device)
    for name, image, mask in zip(names, images, labels):
        image_array = np.asarray(image)
        image_array = image_array.astype(np.float32)
        image_array = image_array / 255

        image_tensor = TF.to_tensor(image_array)

        image_tensor = image_tensor.unsqueeze(0).to(device)

        prediction = model(image_tensor)

        prediction = prediction.squeeze(0)
        
        prediction = prediction.permute(1, 2, 0)
        
        prediction = prediction.squeeze(2)

        prediction = prediction.detach().to("cpu").numpy()
        
        prediction = (prediction > 0.2)

        predicted_image = Image.fromarray(prediction)

        # TODO: save in prediction folder:
        # - image
        # - mask
        # - prediction

        os.mkdir(path + str(name))
        image.save(os.path.join(path, str(name), "image.png"))
        mask.save(os.path.join(path, str(name), "mask.png"))
        predicted_image.save(os.path.join(path, str(name), "prediction.png"))


