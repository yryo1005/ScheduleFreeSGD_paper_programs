import torch
from torch import optim

class ScheduleFreeSGD(optim.Optimizer):
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
            内分パラメータから平均パラメータへ切り替える関数
            推論の直前で使用する
        """
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'r' in state:
                        p.copy_(state['r'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """
            平均パラメータから内分パラメータへ切り替える関数
            学習の直前で使用する
        """
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 's' in state:
                        p.copy_(state['s'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            k = group['k']

            for p in group['params']:
                
                state = self.state[p]
                if 'w' not in state:
                    state['w'] = p.detach().clone() # 真のパラメータ
                    state['r'] = p.detach().clone() # 平均パラメータ
                    state['s'] = p.detach().clone() # 内分パラメータ

                w = state['w']
                r = state['r']
                s = state['s']
                grad = p.grad

                # 真のパラメータを更新
                w.sub_(grad, alpha=lr)
                # 平均パラメータを更新
                tmp = 1 / (k + 1)
                r.mul_(1 - tmp).add_(w, alpha = tmp)
                # 内分パラメータを更新
                s.copy_(r * gamma).add_(w, alpha = 1 - gamma)

                p.copy_(s)

            group['k'] = k + 1