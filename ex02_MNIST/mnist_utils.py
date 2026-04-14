import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import sys
sys.path.append("../")
from utils import *

def load_data(batch_size = 16, test_size = None, num_workers = 1, seed = 0):

    train_dataset = datasets.MNIST(root = "/home/user/workspace/ScheduleFreeSGD_paper/data", train = True, 
                                   transform = transforms.Compose([transforms.ToTensor()]), download = True)
    test_dataset = datasets.MNIST(root = "/home/user/workspace/ScheduleFreeSGD_paper/data", train = False, 
                                  transform = transforms.Compose([transforms.ToTensor()]), download = True)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  generator = generator, 
                                  num_workers = num_workers)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size = batch_size, 
                                 num_workers = num_workers)

    return train_dataloader, test_dataloader

def load_model(seed = 0):

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, inputs):
            inputs = inputs.view(-1, 784)
            Y = self.fc1(inputs)
            Y = torch.relu(Y)
            Y = self.fc2(Y)

            return Y

    set_seed(seed = seed)
    model = MyModel()

    return model

def train(target_path, optimizer, optimizer_params,
          epochs = 100, batch_size = 16, test_size = 0.1, num_workers = 1,
          seed = 0, verbose = False):
    
    print(target_path)

    initial_metrics_sum = {"loss": 0, "acc": 0}
    metrics = [f"train_{M}" for M in initial_metrics_sum.keys()] + \
              [f"test_{M}" for M in initial_metrics_sum.keys()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def loss_func(output, target):
        loss = nn.CrossEntropyLoss()(output, target)
        return loss

    def metric_func(output, target):
        output = output.argmax(dim = 1)
        acc = (output == target).float().mean()
        return acc

    def iteration(X_batch, T_batch, model, optimizer):
        X_batch = X_batch.to(device)
        T_batch = T_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad()
            Y_batch = model(X_batch)
            loss = loss_func(Y_batch, T_batch)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                Y_batch = model(X_batch)
                loss = loss_func(Y_batch, T_batch)

        acc = metric_func(Y_batch, T_batch)

        return {"loss": loss.item(),
                "acc": acc.item()}

    def epoch(dataloader, model, optimizer = None):

        metric_sum = copy.deepcopy(initial_metrics_sum)
        num_samples = 0

        if verbose: pb = tqdm(dataloader)
        else: pb = dataloader

        for X_batch, T_batch in pb:

            metric_batch = iteration(X_batch, T_batch, model, optimizer)

            for K in metric_batch.keys():
                metric_sum[K] += metric_batch[K] * len(X_batch)
            num_samples += len(X_batch)

            if verbose: pb.set_postfix({K:V / num_samples for K, V in metric_sum.items()})

        metric_avg = {K:V / num_samples for K, V in metric_sum.items()}

        return metric_avg

    train_dataloader, test_dataloader = load_data(batch_size = batch_size,
                                                  test_size = test_size,
                                                  num_workers = num_workers,
                                                  seed = seed)
    model = load_model(seed = seed).to(device)
    optimizer = optimizer(model.parameters(), **optimizer_params)

    logger = ResultLogger()
    logger.set_names(*metrics)

    for i in range(epochs + 1):
        if verbose: print(f"epoch; {i}")
        
        # train
        if i == 0:
            model.eval()
            if hasattr(optimizer, "eval"): optimizer.eval()
            train_metric_avg = epoch(train_dataloader, model)
        else:
            model.train()
            if hasattr(optimizer, "train"): optimizer.train()
            train_metric_avg = epoch(train_dataloader, model, optimizer)
        
        # test
        model.eval()
        if hasattr(optimizer, "eval"): optimizer.eval()
        test_metric_avg = epoch(test_dataloader, model)

        logger(*train_metric_avg.values(), *test_metric_avg.values())

    logger.save(target_path)

    del model, optimizer, train_dataloader, test_dataloader
    torch.cuda.empty_cache()