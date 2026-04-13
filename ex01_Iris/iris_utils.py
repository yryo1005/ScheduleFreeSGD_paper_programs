import copy
from tqdm import tqdm

import sklearn.datasets as sk_datasets
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append("../")
from utils import *

def load_data(batch_size = 16, test_size = 0.1, num_workers = 1, seed = 0):

    class IrisDataset(Dataset):
        def __init__(self, inputs, teacher_signals):
            if len(inputs) != len(teacher_signals):
                raise

            self.inputs = inputs
            self.teacher_signals = teacher_signals

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input = torch.tensor( self.inputs[idx], dtype = torch.float32 )
            teacher_signal = torch.tensor(self.teacher_signals[idx], dtype = torch.int64)

            return input, teacher_signal

    def collate_fn(batch):
        inputs = torch.stack( [B[0] for B in batch] )
        teacher_signals = torch.stack( [B[1] for B in batch] )

        return inputs, teacher_signals

    iris = sk_datasets.load_iris()
    inputs = iris.data
    teacher_signals = iris.target

    tmp = list(zip(inputs, teacher_signals))
    train_tmp, test_tmp = train_test_split(tmp, test_size = test_size, random_state = seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_inputs, train_teacher_signals = zip(*train_tmp)
    train_dataset = IrisDataset(train_inputs, train_teacher_signals)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = True,
        generator = generator
    )

    test_inputs, test_teacher_signals = zip(*test_tmp)
    test_dataset = IrisDataset(test_inputs, test_teacher_signals)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        num_workers = num_workers,
        shuffle = False,
    )

    return train_dataloader, test_dataloader

def load_model(seed = 0):

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(4, 3)

        def forward(self, inputs):
            Y = self.fc1(inputs)

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

    def iteration(X_batch, T_batch, model, optimizer = None):
        X_batch = X_batch.to(device)
        T_batch = T_batch.to(device)

        if optimizer is not None:
            model.train()
            if hasattr(optimizer, "train"): optimizer.train()

            optimizer.zero_grad()
            Y_batch = model(X_batch)
            loss = loss_func(Y_batch, T_batch)
            loss.backward()
            optimizer.step()

        else:
            model.eval()
            if hasattr(optimizer, "eval"): optimizer.eval()

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

    for i in range(epochs):
        if verbose: print(f"epoch; {i}")

        if i == 0:
            train_metric_avg = epoch(train_dataloader, model)
        else:
            train_metric_avg = epoch(train_dataloader, model, optimizer)
        test_metric_avg = epoch(test_dataloader, model)

        logger(*train_metric_avg.values(), *test_metric_avg.values())

    logger.save(target_path)

    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()