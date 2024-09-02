import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_images = []
        self.hr_images = sorted([os.path.join(hr_dir, x) for x in os.listdir(hr_dir)])
        self.transform = transform

        # Gather all LR images from subdirectories
        for root, dirs, files in os.walk(lr_dir):
            for file in files:
                self.lr_images.append(os.path.join(root, file))

        self.lr_images = sorted(self.lr_images)

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = Image.open(self.lr_images[idx])
        hr_image = Image.open(self.hr_images[idx])
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # You can adjust this if needed
    transforms.ToTensor()
])

# Paths for training data
train_hr_dir = 'C:\\Users\\Lenovo\\Desktop\\final\\LESRCNN\\data\\train\\HR\\DIV2K_train_HR'
train_lr_dir = 'C:\\Users\\Lenovo\\Desktop\\final\\LESRCNN\\data\\train\\LR\\DIV2K_train_LR_bicubic\\X4'

# Paths for validation data
val_hr_dir = 'C:\\Users\\Lenovo\\Desktop\\final\\LESRCNN\\data\\val\\HR\\DIV2K_valid_HR'
val_lr_dir = 'C:\\Users\\Lenovo\\Desktop\\final\\LESRCNN\\data\\val\\LR\\DIV2K_valid_LR_bicubic\\X4'

# Create datasets
train_dataset = CustomDataset(lr_dir=train_lr_dir, hr_dir=train_hr_dir, transform=transform)
val_dataset = CustomDataset(lr_dir=val_lr_dir, hr_dir=val_hr_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
