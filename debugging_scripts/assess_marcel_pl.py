import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 3)
        self.target = torch.randn(size, 1)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)



# Create dummy datasets and data loaders
train_dataset = DummyDataset(1000)
val_dataset = DummyDataset(200)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create PyTorch Lightning trainer
# Available names are: auto, cpu, cuda, hpu, ipu, mps, tpu.
trainer = pl.Trainer(#accelerator="cuda" if torch.cuda.is_available() else None,
                     max_epochs=5,
                     devices="auto")

# Create the model and train on GPU
model = MyModel()
#trainer.fit(model)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available. Using CPU.")
