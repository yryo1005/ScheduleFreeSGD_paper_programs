import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

class ResultLogger:
    def __init__(self, target_path = None):
        self.names = None
        self.history = {}

        if target_path:
            self.load(target_path)

    def set_names(self, *names):
        if self.names:
            raise

        self.names = list(names)
        for name in self.names:
            if name not in self.history:
                self.history[name] = []

    def __call__(self, *values):
        if self.names is None:
            raise
        if len(values) != len(self.names):
            raise

        for name, value in zip(self.names, values):
            self.history[name].append(value)

    def save(self, target_path):
        with open(target_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def load(self, target_path):
        with open(target_path, "r") as f:
            data = json.load(f)
            self.history = data
            self.names = list(data.keys())

    def __getitem__(self, key):
        return self.history.get(key, [])

def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_param_id(params):
    """
        params; パラメータ (ex {"lr": 0.01, "momentum": 0.9}
    """
    return "_".join([str(params[k]) for k in sorted(params.keys())])

def grid_search(target_dir, optimizer, fixed_params, search_space, train,
                epochs, batch_size, test_size, num_seed,
                num_workers = 1, max_parallel = None, verbose = False):
    """
        target_dir; 保存先のディレクトリ
        optimizer; torch.optim.Optimizer
        fixed_params; 固定のハイパーパラメータ (ex {"dampening": 0, "weight_decay": 0, "nesterov": False}
        search_space; ハイパーパラメータの探索空間 (ex {"lr": [0.1, 0.01], "momentum": [0.0, 0.9]}
        train; 学習関数
    """
    os.makedirs(target_dir, exist_ok = True)
    for PARAMS in product(*search_space.values()):

        # 探索空間のパラメータ
        params = {
            K: P for K, P in zip(search_space.keys(), PARAMS)
        }

        for seed in range(num_seed):

            param_id = get_param_id(params)
            target_path = f"{target_dir}/{param_id}_{seed}.json"

            # 最適化手法のパラメータ
            optimizer_params = {
                **params,
                **fixed_params
            }
            if os.path.exists(target_path):
                continue

            train(target_path = target_path,
                  optimizer = optimizer,
                  optimizer_params = optimizer_params,
                  seed = seed,
                  epochs = epochs,
                  batch_size = batch_size,
                  test_size = test_size,
                  num_workers = num_workers,
                  verbose = verbose
                )

        with open(f"{target_dir}/{param_id}_config.json", "w") as f:
            json.dump(optimizer_params, f, indent = 4)

# グリッドサーチをマルチプロセスで実行する関数
# デバック時はコメントアウト
def grid_search(target_dir, optimizer, fixed_params, search_space, train,
                epochs, batch_size, test_size, num_seed, 
                num_workers = 1, verbose = False, max_parallel = 2):
    
    os.makedirs(target_dir, exist_ok = True)
    
    tasks = []
    for PARAMS in product(*search_space.values()):
        params = {K: P for K, P in zip(search_space.keys(), PARAMS)}
        param_id = get_param_id(params)
        
        optimizer_params = {**params, **fixed_params}
        with open(f"{target_dir}/{param_id}_config.json", "w") as f:
            json.dump(optimizer_params, f, indent=4)

        for seed in range(num_seed):
            target_path = f"{target_dir}/{param_id}_{seed}.json"
            
            if os.path.exists(target_path):
                continue
            
            # trainの引数
            task_args = {
                "target_path": target_path,
                "optimizer": optimizer,
                "optimizer_params": optimizer_params,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "test_size": test_size,
                "num_workers": num_workers,
                "verbose": verbose
            }
            tasks.append(task_args)

    with ProcessPoolExecutor(max_workers = max_parallel) as executor:
        futures = [executor.submit(train, **t) for t in tasks]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed with error: {e}")

def get_best_results(target_dir, search_space, num_seed,
                     target_metric = "test_acc", mode = "max"):

    best_results = dict()
    for PARAMS in product(*search_space.values()):

        # 探索空間のパラメータ
        params = {
            K: P for K, P in zip(search_space.keys(), PARAMS)
        }

        best_metric = None
        for seed in range(num_seed):

            param_id = get_param_id(params)
            target_path = f"{target_dir}/{param_id}_{seed}.json"
            logger = ResultLogger(target_path = target_path)

            if best_metric is None:
                best_metric = logger[target_metric][-1]
            else:
                if mode == "min":
                    best_metric = min(best_metric, logger[target_metric][-1])
                elif mode == "max":
                    best_metric = max(best_metric, logger[target_metric][-1])
            if best_metric == logger[target_metric][-1]:
                best_results[param_id] = logger

    return best_results

def plot_best_training_results(target_dir, best_results, search_space, metrics):

    fig = plt.figure(figsize = (10, 10))
    for i, M in enumerate(metrics):
        num_metrics = len(metrics)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        ax = fig.add_subplot(rows, cols, i + 1)

        colors = [plt.get_cmap('turbo')(i) for i in np.linspace(0, 1, len(best_results))]
        for K, V in best_results.items():
            ax.plot(V[M], label = f"{K}", color = colors.pop())
        ax.set_xlabel("epoch")
        ax.set_ylabel(M)
        ax.set_title(M)
        ax.grid()

    tmp = ""
    for K in sorted(search_space.keys()):
        tmp += f"{K}_"
    tmp = tmp[:-1]
    fig.suptitle(tmp)

    fig.legend(
        labels=[K for K in best_results.keys()],
        loc='upper center',
        ncol=5,  # 横に並べる列数
        bbox_to_anchor=(0.5, 1.05) # figure の外に配置
    )

    fig.savefig(f"{target_dir}/best.png", bbox_inches = "tight")
    plt.show()

def get_confidence_intervals(target_dir, search_space, metrics, num_seed):
    confidence_intervals = dict()

    for PARAMS in product(*search_space.values()):

        # 探索空間のパラメータ
        params = {
            K: P for K, P in zip(search_space.keys(), PARAMS)
        }

        histories = {M: list() for M in metrics}
        for seed in range(num_seed):
            param_id = get_param_id(params)
            target_path = f"{target_dir}/{param_id}_{seed}.json"
            logger = ResultLogger(target_path = target_path)
            for M in metrics:
                histories[M].append(logger[M])
            confidence_intervals[param_id] = histories

        confidence_intervals[param_id] = {
            M: {
                "mean": np.array(histories[M]).mean(axis = 0),
                "std": np.array(histories[M]).std(axis = 0)
            }
            for M in metrics
        }

    return confidence_intervals

def plot_confidence_intervals(target_dir, confidence_intervals, metrics, ):

    for K, V in confidence_intervals.items():

        fig = plt.figure(figsize = (10, 10))
        for i, M in enumerate(metrics):
            num_metrics = len(metrics)
            cols = 2
            rows = (num_metrics + cols - 1) // cols
            ax = fig.add_subplot(rows, cols, i + 1)

            ax.plot(V[M]["mean"], label = f"{V[M]['mean'][-1]:.4f} ± {V[M]['std'][-1]:.4f}")
            ax.fill_between(
                x = range(len(V[M]["mean"])),
                y1 = V[M]["mean"] - V[M]["std"],
                y2 = V[M]["mean"] + V[M]["std"],
                alpha=0.2,
            )

            ax.set_xlabel("epoch")
            ax.set_ylabel(M)
            ax.set_title(M)
            ax.legend()
            ax.grid()
        fig.suptitle(K)
        fig.savefig(f"{target_dir}/{K}_confidence_intervals.png", bbox_inches = "tight")

        plt.show()

def run(target_dir, optimizer, fixed_params, search_space, train,
        epochs, batch_size, test_size, num_workers,
        num_seed = 5, max_parallel = 1, verbose = False,
        metrics = ["train_loss", "train_acc", "test_loss", "test_acc"], target_metric = "test_acc", mode = "max"):
    """
        target_dir; 保存先のディレクトリ
        optimizer; torch.optim.Optimizer
        fixed_params; 固定のハイパーパラメータ (ex {"dampening": 0, "weight_decay": 0, "nesterov": False}
        search_space; ハイパーパラメータの探索空間 (ex {"lr": [0.1, 0.01], "momentum": [0.0, 0.9]}
        train; 学習関数
    """

    grid_search(target_dir = target_dir,
                optimizer = optimizer,
                fixed_params = fixed_params,
                search_space = search_space,
                train = train,
                epochs = epochs,
                batch_size = batch_size,
                test_size = test_size,
                num_workers = num_workers,
                num_seed = num_seed,
                max_parallel = max_parallel,
                verbose = verbose)

    best_results = get_best_results(target_dir = target_dir,
                                    search_space = search_space,
                                    target_metric = target_metric,
                                    mode = mode,
                                    num_seed = num_seed)

    plot_best_training_results(target_dir = target_dir,
                               best_results = best_results,
                               search_space = search_space,
                               metrics = metrics)

    confidence_intervals = get_confidence_intervals(target_dir = target_dir,
                                                    search_space = search_space,
                                                    metrics = metrics,
                                                    num_seed = num_seed)

    plot_confidence_intervals(target_dir = target_dir,
                              confidence_intervals = confidence_intervals,
                              metrics = metrics)

    training_results = dict()
    for K in confidence_intervals.keys():
        training_results[K] = dict()
        for M in confidence_intervals[K].keys():
            training_results[K][M] = {
                "best": best_results[K][M],
                "mean": confidence_intervals[K][M]["mean"].tolist(),
                "std": confidence_intervals[K][M]["std"].tolist()
            }

        with open(f"{target_dir}/{K}.json", "w") as f:
            json.dump(training_results[K], f, indent = 4)

    with open(f"{target_dir}/training_results.json", "w") as f:
        json.dump(training_results, f, indent=4)

def run_all(root_dir, optimizer_to_params, train,
            epochs, batch_size, test_size, num_seed, 
            num_workers = 1, max_parallel = None, verbose = False,
            metrics = ["train_loss", "train_acc", "test_loss", "test_acc"], 
            target_metric = "test_acc", mode = "max"):
    """
        target_dir; 保存先のディレクトリ
        optimizer; torch.optim.Optimizer
        fixed_params; 固定のハイパーパラメータ (ex {"dampening": 0, "weight_decay": 0, "nesterov": False}
        search_space; ハイパーパラメータの探索空間 (ex {"lr": [0.1, 0.01], "momentum": [0.0, 0.9]}
        train; 学習関数
    """

    root_dir = f"{root_dir}/{epochs}_{batch_size}_{test_size}"
    os.makedirs(root_dir, exist_ok = True)

    optimizer_to_results = dict()

    for K, V in optimizer_to_params.items():
        target_dir = f"{root_dir}/{K}"

        run(target_dir = target_dir,
            optimizer = V["optimizer"],
            fixed_params = V["fixed_params"],
            search_space = V["search_space"],
            train = train,
            epochs = epochs,
            batch_size = batch_size,
            test_size = test_size,
            num_seed = num_seed,
            num_workers = num_workers,
            max_parallel = max_parallel,
            verbose = verbose,
            metrics = metrics,
            target_metric = target_metric,
            mode = mode)

        with open(f"{target_dir}/training_results.json", "r") as f:
            optimizer_to_results[K] = json.load(f)

    optimizer_to_best_results = dict()
    for K in optimizer_to_params.keys():
        with open(f"{root_dir}/{K}/training_results.json", "r") as f:
            results = json.load(f)

        best = None
        for KK, VV in results.items():
            if best == None:
                best = {"params": KK, **VV}
            else:
                if mode == "max":
                    if best[target_metric]["mean"][-1] < VV[target_metric]["mean"][-1]:
                        best = {"params": KK, **VV}
                elif mode == "min":
                    if best[target_metric]["mean"][-1] > VV[target_metric]["mean"][-1]:
                        best = {"params": KK, **VV}

        tmp = f"{K}_{best['params']}"
        optimizer_to_best_results[tmp] = best

    fig = plt.figure(figsize = (10, 10))
    for i, M in enumerate(metrics):
        num_metrics = len(metrics)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        ax = fig.add_subplot(rows, cols, i + 1)

        colors = [plt.get_cmap('turbo')(i) for i in np.linspace(0, 1, len(optimizer_to_best_results))]

        for K, V in optimizer_to_best_results.items():
            color = colors.pop()
            ax.plot(V[M]["mean"], label = f"{K} ± {V[M]['std'][-1]:.4f}", color = color)
            ax.fill_between(
                x = range(len(V[M]["mean"])),
                y1 = np.array(V[M]["mean"]) - np.array(V[M]["std"]),
                y2 = np.array(V[M]["mean"]) + np.array(V[M]["std"]),
                alpha=0.2,
                color = color
            )

        ax.set_xlabel("epoch")
        ax.set_ylabel(M)
        ax.set_title(M)
        ax.legend()
        ax.grid()

    fig.savefig(f"{root_dir}/best_confidence_intervals.png", bbox_inches = "tight")
    plt.show()

    with open(f"{root_dir}/config.json", "w") as f:
        json.dump({
            "epochs": epochs,
            "batch_size": batch_size,
            "test_size": test_size,
            "num_seed": num_seed,
            "num_workers": num_workers,
            "max_parallel": max_parallel,
        }, f, indent = 4)