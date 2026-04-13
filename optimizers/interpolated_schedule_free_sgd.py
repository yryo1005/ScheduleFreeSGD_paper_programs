import torch
from torch import optim

class InterpolatedScheduleFreeSGD(optim.Optimizer):
    def __init__(
            self,
            params,
            lr = 0.01,
            gamma = 0.9,
    ):
        """
            0 < lr; 学習率
            0 <= gamme <= 1; 真のパラメータと平均パラメータの内分の割合
        """

        defaults = dict(
            lr = lr,
            gamma = gamma,
            k = 0,
            train_mode = True,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        """
            真のパラメータから平均パラメータへ切り替える関数
            推論の直前で使用する
        """
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'w' not in state:
                        state['w'] = p.detach().clone() # 平均パラメータ
                    if 'r' in state:
                        p.copy_(state['r'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """
            平均パラメータから真のパラメータへ切り替える関数
            学習の直前で使用する
        """
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'w' in state:
                        p.copy_(state['w'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            k = group['k']

            for p in group['params']:
                
                state = self.state[p]
                if 'r' not in state:
                    state['r'] = p.detach().clone() # 平均パラメータ
                    state['avg_grad'] = torch.zeros_like(p) # 勾配の平均

                r = state['r']
                avg_grad = state['avg_grad']
                grad = p.grad

                interpolated_grad = gamma * grad + (1 - gamma) * avg_grad

                # 真のパラメータを更新
                p.sub_(interpolated_grad, alpha=lr)
                # 平均パラメータを更新
                tmp = 1 / (k + 1)
                r.mul_(1 - tmp).add_(p, alpha = tmp)
                # 勾配の平均を更新
                avg_grad.mul_(1 - tmp).add_(grad, alpha = tmp)

            group['k'] = k + 1