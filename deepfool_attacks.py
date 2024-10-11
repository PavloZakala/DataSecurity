import numpy as np
import copy

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm

from train_simple_cifar10 import ResNetLightningModel

from fgsm_attack import Accuracy

# https://github.com/aminul-huq/DeepFool/blob/master/DeepFool.ipynb
# https://arxiv.org/pdf/1511.04599

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=10):
    f_image = net.forward(image).data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = torch.tensor(pert_image[None, :], requires_grad=True)

    fs = net.forward(x[0])
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_classes):

            # x.zero_grad()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = torch.tensor(pert_image, requires_grad=True)
        fs = net.forward(x[0])
        k_i = np.argmax(fs.data.numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the model
    model = ResNetLightningModel.load_from_checkpoint('checkpoints/best-checkpoint-v1.ckpt')
    model.model.eval()
    model = model.to('cpu')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    dataloader = DataLoader(trainset, batch_size=1, shuffle=False)


    acc = Accuracy()
    i = 0
    for data, target in tqdm.tqdm(dataloader):
        if i == 100:
            break

        i += 1

        data.requires_grad = True

        r, loop_i, label_orig, label_pert, pert_image = deepfool(data, model.model)

        output = model(pert_image[0])
        acc.update(output, target)

    print(f'Accuracy {acc.compute()}')

