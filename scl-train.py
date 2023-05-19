import os
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import argparse

num_classes = 10
num_workers = 4
device = "cuda"

def get_cifar10(data_aug=False):
    if data_aug:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
                ),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4922, 0.4832, 0.4486], [0.2456, 0.2419, 0.2605]
            ),
        ]
    )
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    n_samples = len(dataset)
    
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), n_samples - int(n_samples*0.9)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    trainset.dataset.ord_labels = deepcopy(trainset.dataset.targets)
    validset.dataset.ord_labels = deepcopy(validset.dataset.targets)
    
    T = torch.full([num_classes, num_classes], 1/(num_classes-1))
    for i in range(num_classes):
        T[i][i] = 0
    
    for i in range(n_samples):
        ord_label = trainset.dataset.targets[i]
        trainset.dataset.targets[i] = np.random.choice(list(range(10)), p=T[ord_label])
    
    for i in range(n_samples):
        ord_label = validset.dataset.targets[i]
        validset.dataset.targets[i] = np.random.choice(list(range(10)), p=T[ord_label])
    
    return trainset, validset, testset, ord_trainset, ord_validset

def get_cifar20(data_aug=False):

    if data_aug:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
                ),
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5068, 0.4854, 0.4402], [0.2672, 0.2563, 0.2760]
            ),
        ]
    )
    
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    n_samples = len(dataset)
    global num_classes
    num_classes = 20

    def _cifar100_to_cifar20(target):
        _dict = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 18, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
        return _dict[target]
    
    dataset.targets = [_cifar100_to_cifar20(i) for i in dataset.targets]
    testset.targets = [_cifar100_to_cifar20(i) for i in testset.targets]
    ord_trainset, ord_validset = torch.utils.data.random_split(dataset, [int(n_samples*0.9), n_samples - int(n_samples*0.9)])
    
    trainset = deepcopy(ord_trainset)
    validset = deepcopy(ord_validset)
    trainset.dataset.ord_labels = deepcopy(trainset.dataset.targets)
    validset.dataset.ord_labels = deepcopy(validset.dataset.targets)
    
    T = torch.full([num_classes, num_classes], 1/(num_classes-1))
    for i in range(num_classes):
        T[i][i] = 0
    
    for i in range(n_samples):
        ord_label = trainset.dataset.targets[i]
        trainset.dataset.targets[i] = np.random.choice(list(range(20)), p=T[ord_label])
    
    for i in range(n_samples):
        ord_label = validset.dataset.targets[i]
        validset.dataset.targets[i] = np.random.choice(list(range(20)), p=T[ord_label])
    
    return trainset, validset, testset, ord_trainset, ord_validset

def get_resnet18():
    resnet = torchvision.models.resnet18(weights=None)
    # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def get_modified_resnet18():
    resnet = torchvision.models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet

def validate(model, dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def get_dataset_T(dataset):
    dataset_T = np.zeros((num_classes,num_classes))
    class_count = np.zeros(num_classes)
    for i in range(len(dataset)):
        dataset_T[dataset.dataset.ord_labels[i]][dataset.dataset.targets[i]] += 1
        class_count[dataset.dataset.ord_labels[i]] += 1
    for i in range(num_classes):
        dataset_T[i] /= class_count[i]
    return dataset_T

def train(args):
    dataset_name = args.dataset_name
    algo = args.algo
    model = args.model
    lr = args.lr
    seed = args.seed
    data_aug = True if args.data_aug.lower()=="true" else False
    eval_n_epoch = args.evaluate_step
    batch_size = args.batch_size
    epochs = args.n_epoch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if data_aug:
        print("Use data augmentation.")

    if dataset_name == "cifar10":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar10(data_aug)
    elif dataset_name == "cifar20":
        trainset, validset, testset, ord_trainset, ord_validset = get_cifar20(data_aug)
    else:
        raise NotImplementedError

    # Print the complementary label distribution T
    dataset_T = get_dataset_T(trainset)
    np.set_printoptions(floatmode='fixed', precision=2)
    print("Dataset's transition matrix T:")
    print(dataset_T)
    dataset_T = torch.tensor(dataset_T, dtype=torch.float).to(device)

    # Set Q for forward algorithm
    Q = dataset_T

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ord_trainloader = DataLoader(ord_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ord_validloader = DataLoader(ord_validset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if args.model == "resnet18":
        model = get_resnet18().to(device)
    elif args.model == "m-resnet18":
        model = get_modified_resnet18().to(device)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    with tqdm(range(epochs), unit="epoch") as tepoch:
        # tepoch.set_description(f"lr={lr}")
        for epoch in tepoch:
            training_loss = 0.0
            model.train()

            for inputs, labels in trainloader:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                    
                if algo == "scl-exp":
                    outputs = F.softmax(outputs, dim=1)
                    loss = -F.nll_loss(outputs.exp(), labels)
                    loss.backward()
                
                elif algo == "scl-fwd":
                    q = torch.mm(F.softmax(outputs, dim=1), Q) + 1e-6
                    loss = F.nll_loss(q.log(), labels.squeeze())
                    loss.backward()
                
                elif algo == "scl-nl":
                    p = (1 - F.softmax(outputs, dim=1) + 1e-6).log()
                    loss = F.nll_loss(p, labels)
                    loss.backward()

                else:
                    raise NotImplementedError
                
                optimizer.step()
                training_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

            training_loss /= len(trainloader)
            
            if (epoch+1) % eval_n_epoch == 0:
                model.eval()
                train_acc, valid_acc, test_acc = validate(model, ord_trainloader), validate(model, ord_validloader), validate(model, testloader)
                print("Accuracy(train/valid/test)", train_acc, valid_acc, test_acc)

if __name__ == "__main__":

    dataset_list = [
        "cifar10",
        "cifar20"
    ]

    algo_list = [
        "scl-exp",
        "scl-nl",
        "scl-fwd"
    ]

    model_list = [
        "resnet18",
        "m-resnet18"
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, choices=algo_list, help='Algorithm')
    parser.add_argument('--dataset_name', type=str, choices=dataset_list, help='Dataset name', default='cifar10')
    parser.add_argument('--model', type=str, choices=model_list, help='Model name', default='resnet18')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--seed', type=int, help='Random seed', default=1126)
    parser.add_argument('--data_aug', type=str, default='false')
    parser.add_argument('--evaluate_step', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    train(args)
