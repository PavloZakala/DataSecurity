import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super(CIFAR10DataModule, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download data (done only once)
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        # Data augmentation and normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Transforms for train/validation datasets
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)


class ResNetLightningModel(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(ResNetLightningModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.data).item() / len(labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Using SGD optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # Alternatively, you can use Adam optimizer
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == '__main__':
    # Initialize the model
    model = ResNetLightningModel()

    # Initialize the data module
    data_module = CIFAR10DataModule()

    # Model checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback]
    )

    # Train the model
    # trainer.fit(model, data_module)

    # # Test the model on the test set
    # trainer.test(model, data_module)

    # Load the best checkpoint
    best_model = ResNetLightningModel.load_from_checkpoint('./checkpoints/best-checkpoint-v1.ckpt')

    # Move the model to evaluation mode
    best_model.eval()

    # Test the model on the test set
    trainer.validate(best_model, data_module)
