import numpy as np
import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from fgsm_attack import Accuracy
from train_simple_cifar10 import ResNetLightningModel

class AlbumentationsTransform:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform

    def __call__(self, img):
        # Convert PIL image to NumPy array
        img = np.array(img)
        # Apply Albumentations transformation
        augmented = self.albumentations_transform(image=img)['image']
        return augmented

    def __str__(self):
        return self.albumentations_transform.__str__()

if __name__ == '__main__':

    transforms_test = [
        transforms.Compose([ AlbumentationsTransform(A.Compose([
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(),]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Rotate(limit=40, p=0.5),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.ToGray(p=0.2),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Rotate(limit=40, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Blur(blur_limit=3, p=0.5),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.RandomSunFlare(),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(), ]
        ))]),

        ]

    # Load the model
    model = ResNetLightningModel.load_from_checkpoint('checkpoints/best-checkpoint-v1.ckpt')
    model.model.eval()

    for transform in transforms_test:

        acc = Accuracy()

        trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        dataloader = DataLoader(trainset, batch_size=64, shuffle=False)

        for data, target in tqdm.tqdm(dataloader):

            data = data.to('cuda')
            target = target.to('cuda')

            output = model(data)

            acc.update(output, target)

        print(f'Accuracy {acc.compute()}, transforms:  {transform}')

