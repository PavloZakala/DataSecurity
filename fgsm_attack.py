import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchmetrics import Metric
import tqdm

from train_simple_cifar10 import ResNetLightningModel

class Accuracy(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target).cpu()
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total.float()

    def reset(self):
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

# Fast Gradient Sign Method
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -2.5, 2.5)
    return perturbed_image


if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the model
    model = ResNetLightningModel.load_from_checkpoint('checkpoints/best-checkpoint-v1.ckpt')
    model.model.eval()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    dataloader = DataLoader(trainset, batch_size=64, shuffle=False)


    for eps in torch.arange(0.0, 0.5, 0.05):

        acc = Accuracy()
        total_loss = 0
        # i = 0
        for data, target in tqdm.tqdm(dataloader):

            data = data.to('cuda')
            target = target.to('cuda')
            data.requires_grad = True

            output = model(data)
            loss = nn.functional.cross_entropy(output, target)

            model.zero_grad()
            loss.backward()

            data_grad = data.grad.data

            perturbed_data = fgsm_attack(data, eps, data_grad)

            output = model(perturbed_data)
            loss = nn.functional.cross_entropy(output, target)
            total_loss += loss.detach().cpu().item()

            final_pred = output.max(1, keepdim=True)[1]
            acc.update(output, target)

        print(f'Accuracy for epsilon {eps:.2f}: {acc.compute()}, {total_loss/len(dataloader)}')

