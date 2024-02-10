import os

from PIL import Image


def get_all_images() -> list[list[Image], list[Image], list[Image]]:
    path = "./0.1 - base/"

    images_names = os.listdir(path)

    images_w_1920 = []
    images_w_1280 = []
    images_w_1024 = []

    for i in images_names:

        img = Image.open(os.path.join(path, i))

        img = img.convert("L")

        if img.width == 1920:
            images_w_1920.append(img)

        if img.width == 1280:
            images_w_1280.append(img)

        if img.width == 1024:
            images_w_1024.append(img)

    return [images_w_1920, images_w_1280, images_w_1024]


def crop_all_images_to_1280x876(images: list[Image]) -> list[Image]:
    cropped = []

    for image in images:
        h = image.height  # 1920
        w = image.width  # 1080

        left = (w - 1280) / 2  # 320
        top = (h - 876) / 2  # 102
        right = w - left
        bottom = h - top

        new_image = image.crop((int(left), int(top), int(right), int(bottom)))

        cropped.append(new_image)

    return cropped


def expand_to_1280x876(images: list[Image]):
    padded = []
    background_color = 0

    for image in images:
        h = image.height
        w = image.width

        left = (1280 - w) / 2
        top = (876 - h) / 2  # 156
        right = 1280 - left
        bottom = 876 - top

        new_image = Image.new(image.mode, (1280, 876), background_color)
        new_image.paste(image, (int(left), int(top), int(right), int(bottom)))

        padded.append(new_image)

    return padded


def save_pad_and_crop() -> None:
    path = "./0.2 - padded and cropped/"

    images_name = os.listdir(path)

    for i in images_name:
        os.remove(os.path.join(path, i))

    images_w_1920, images_w_1280, images_w_1024 = get_all_images()

    images = (
        expand_to_1280x876(images_w_1024)
        + images_w_1280
        + crop_all_images_to_1280x876(images_w_1920)
    )

    print(len(images))

    for index, (image) in enumerate(images):
        print(image.height)
        print(image.width)

        image.save(str(path + str(index) + ".png"), format="png")


def get_pad_and_crop() -> list[Image]:
    images_w_1920, images_w_1280, images_w_1024 = get_all_images()

    images = (
        expand_to_1280x876(images_w_1024)
        + images_w_1280
        + crop_all_images_to_1280x876(images_w_1920)
    )

    return images


def crop_image_y_border() -> list[Image]:
    images = get_pad_and_crop()

    cropped = []

    for image in images:
        h = image.height  # 876
        w = image.width  # 1280

        left = 0
        top = h * 0.10  # 102
        right = w - left
        bottom = h - top

        new_image = image.crop((int(left), int(top), int(right), int(bottom)))

        cropped.append(new_image)

    return cropped


if __name__ == "__main__":
    save_pad_and_crop()
