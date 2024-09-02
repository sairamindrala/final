import torch
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, val_loader  # Importing from the dataset.py file
from model import LESRCNN  # Assuming LESRCNN is defined in model.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss, optimizer
model = LESRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for lr_images, hr_images in train_loader:
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad()
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    # Validation step (optional)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for lr_images, hr_images in val_loader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            val_loss += loss.item()
    
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

    # Save model checkpoint
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
