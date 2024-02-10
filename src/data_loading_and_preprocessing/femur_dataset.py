from torch.utils.data import Dataset



class FemurDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image, label

    def __len__(self):
        return len(self.images)
