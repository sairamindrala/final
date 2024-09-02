import torch
from dataset import val_loader  # Importing from the dataset.py file
from model import LESRCNN  # Importing the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = LESRCNN().to(device)
model.load_state_dict(torch.load('model_epoch_50.pth'))  # Adjust the model file name as needed
model.eval()

# Evaluate on the validation set
with torch.no_grad():
    for i, (lr_images, hr_images) in enumerate(val_loader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        outputs = model(lr_images)
        
        # Example: Save or display the first image from the batch
        if i == 0:
            torchvision.utils.save_image(outputs[0], 'output.png')
