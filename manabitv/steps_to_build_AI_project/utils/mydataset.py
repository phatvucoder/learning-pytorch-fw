from torch.utils import data
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        
        if img_path.split("/")[-2] == "ants":
            label = 0
        elif img_path.split("/")[-2] == "bees":
            label = 1
        
        return img_transformed, label