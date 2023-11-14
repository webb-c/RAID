import torch
import numpy as np
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

import advertorch.attacks as attacks
from PIL import Image

# adv dataset
def make_adv_img(model, model_name):
    device ='cuda'
    model = model.to(device).eval()
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))]
    )
    img_transform = transforms.ToPILImage()

    trainset = datasets.CIFAR10(root='./data/cifar10/', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform)

    data_loader = DataLoader(trainset, 1, shuffle=True, num_workers=1)

    file = []
    adv_file = []
    file_name = []

    at = attacks.LinfPGDAttack(model, eps=6.0/255, nb_iter=7)

    for x, y in tqdm(data_loader, 'images'):
        x = x.to(device)
        y = y.to(device)
        logit = model(x)
        pred = torch.nn.functional.softmax(logit, dim=1)
        output = torch.argmax(pred, dim=1)
        if y != output:
            continue

        advx = at.perturb(x, y)
        logit_adv = model(advx)
        pred_adv = torch.nn.functional.softmax(logit_adv, dim=1)
        output_adv = torch.argmax(pred_adv, dim=1)
        if y == output_adv:
            continue

        file.append(x.squeeze().cpu())
        adv_file.append(advx.squeeze().cpu())
        file_name.append((output.item(), output_adv.item()))

    data_loader = DataLoader(testset, 1, shuffle=True, pin_memory=True)
    for x, y in tqdm(data_loader, 'images'):
        x = x.to(device)
        y = y.to(device)
        logit = model(x)
        pred = torch.nn.functional.softmax(logit, dim=1)
        output = torch.argmax(pred, dim=1)
        if y != output:
            continue

        advx = at.perturb(x, y)
        logit_adv = model(advx)
        pred_adv = torch.nn.functional.softmax(logit_adv, dim=1)
        output_adv = torch.argmax(pred_adv, dim=1)
        if y == output_adv:
            continue

        file.append(x.squeeze().cpu())
        adv_file.append(advx.squeeze().cpu())
        file_name.append((output.item(), output_adv.item()))
    
    train_size = int(0.6*len(file))
    val_size = int(0.2*len(file))

    for idx, (x, advx, name) in enumerate(zip(file[:train_size], adv_file[:train_size], file_name[:train_size])):
        img_transform(x).save(f"./images/{model_name}/origin/train/{idx}_{name[0]}.png")
        img_transform(advx).save(f"./images/{model_name}/adv/train/{idx}_{name[0]}_{name[1]}.png")

    for idx, (x, advx, name) in enumerate(zip(file[train_size:train_size+val_size], adv_file[train_size:train_size+val_size], file_name[train_size:train_size+val_size])):
        img_transform(x).save(f"./images/{model_name}/origin/val/{idx}_{name[0]}.png")
        img_transform(advx).save(f"./images/{model_name}/adv/val/{idx}_{name[0]}_{name[1]}.png")

    for idx, (x, advx, name) in enumerate(zip(file[train_size+val_size:], adv_file[train_size+val_size:], file_name[train_size+val_size:])):
        img_transform(x).save(f"./images/{model_name}/origin/test/{idx}_{name[0]}.png")
        img_transform(advx).save(f"./images/{model_name}/adv/test/{idx}_{name[0]}_{name[1]}.png")

def load_model(model_name, path=None):
    if model_name == 'mobilenet':
        model = mobilenet_v2(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier._modules["1"].in_features
        model.classifier._modules["1"] = torch.nn.Linear(num_ftrs, 10)
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        if path is not None:
            model.load_state_dict(torch.load(path))

    return model

def pretrain_model(model_name='mobilenet'):
    device = 'cuda'
    model = load_model(model_name)
    model = model.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0, 0, 0), (1, 1, 1))]
    )

    b = 32
    
    train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size
    trainset, validset = random_split(train_dataset, [train_size, valid_size])
    
    trainloader = DataLoader(trainset, b, shuffle=True, num_workers=1)
    validloader = DataLoader(validset, b, shuffle=True, num_workers=1)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch = 30

    for e in range(epoch):
        model.train()
        train_loss = 0
        train_correct_count = 0
        for x, y in tqdm(trainloader, 'train'):
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            train_correct_count += (outputs == y).sum().item()

            loss = torch.nn.functional.cross_entropy(logit, y)
            train_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss = train_loss / len(trainset)
        train_acc = train_correct_count / len(trainset)

        if e%5==0:
            model.eval()
            val_loss = 0
            val_correct_count = 0
            for x, y in tqdm(validloader, 'val'):
                x = x.to(device)
                y = y.to(device)
                logit = model(x)
                pred = torch.nn.functional.softmax(logit, dim=1)
                outputs = torch.argmax(pred, dim=1)
                val_correct_count += (outputs == y).sum().item()

                loss = torch.nn.functional.cross_entropy(logit, y)
                val_loss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
            val_loss = val_loss / len(validset)
            val_acc = val_correct_count / len(validset)

            print(f"epoch {e}\ntrain_acc: {train_acc}\tloss: {train_loss}\nval_acc: {val_acc}\tval_loss: {val_loss}")

    torch.save(model.state_dict(), f"./models/{model_name}.pt")

    return model

def test_image():
    model = load_model('mobilenet', './models/mobilenet.pt')
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0, 0, 0), (1, 1, 1))]
    )

    ori_img_path = './images/mobilenet/origin/train/0_9.png'
    adv_img_path = './images/mobilenet/adv/train/0_9_1.png'

    with Image.open(ori_img_path) as img:
        x = transform(img)
    with Image.open(adv_img_path) as img:
        advx = transform(img)

    model.eval()
    logit = model(x.unsqueeze(0))
    pred = torch.nn.functional.softmax(logit, dim=1)
    outputs = torch.argmax(pred, dim=1)
    print('clean image classification:','9', outputs.item())

    logit = model(advx.unsqueeze(0))
    pred = torch.nn.functional.softmax(logit, dim=1)
    outputs = torch.argmax(pred, dim=1)
    print('adv image classification:','1', outputs.item())


if __name__ == "__main__":
    print('Execute Adversarial.py')

    model_name = 'mobilenet'
    # pretrain_model()
    model = load_model(model_name, './models/mobilenet.pt')
    make_adv_img(model, model_name)
    # test_image()