from random import shuffle
import torch
from src.PCFGAN.PCFGAN import char_func_path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
# from src.evaluations.test_metrics import Sig_mmd, Sig_mmd_small
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# import signatory
# from src.evaluations.evaluate import _train_regressor
from src.utils import to_numpy
from src.baselines.RCF_GAN import CFLossFunc
import matplotlib.pyplot as plt
import seaborn as sns


def optimize_CF(
    X_dl: torch.tensor,
    Y_dl: torch.tensor,
    iterations: int,
    device,
    M_num_samples: int = 8,
    hidden_size: int = 8,
    input_size: int = 2,
    lie_group: str = 'upper'
):
    char_func = char_func_path(
        num_samples=M_num_samples,
        hidden_size=hidden_size,
        input_size=input_size,
        lie_group=lie_group,
        add_time=True,
        init_range=1,
    ).to(device)
    best_loss = 0.
    char_optimizer = torch.optim.Adam(char_func.parameters(), betas=(0, 0.9), lr=0.002)
    print("start opitmize charateristics function")
    for i in tqdm(range(iterations)):
        char_func.train()
        char_optimizer.zero_grad()
        X = next(iter(X_dl))
        Y = next(iter(Y_dl))
        char_loss = -char_func.distance_measure(X, Y, Lambda=0)
        if -char_loss > best_loss:
            print("Loss updated: {}".format(-char_loss))
            best_loss = -char_loss
        # print(char_loss)
        # char_loss = - self.char_func.distance_measure(
        #   self.D(x_real), self.D(x_fake))
        char_loss.backward()
        char_optimizer.step()

    trained_char_func = char_func
    return trained_char_func


class Compare_test_metrics:
    def __init__(self, X, Y, config):
        self.X = X
        self.Y = Y
        self.config = config

    def permutation_test(self, test_func, num_perm, sample_size):
        with torch.no_grad():
            # X = self.subsample(self.X, sample_size)
            # Y = self.subsample(self.Y, sample_size)
            # X = X.to(self.config.device)
            # Y = Y.to(self.config.device)
            #
            # # print(t1)
            # n, m = X.shape[0], Y.shape[0]
            # combined = torch.cat([X, Y])
            H0_stats = []
            H1_stats = []
            H2_stats = []

            for i in tqdm(range(num_perm)):
                X = self.subsample(self.X, sample_size)
                Y = self.subsample(self.Y, sample_size)
                X = X.to(self.config.device)
                Y = Y.to(self.config.device)
                n, m = X.shape[0], Y.shape[0]
                combined = torch.cat([X, Y])
                idx = torch.randperm(n + m)
                H0_stats.append(
                    test_func(combined[idx[:n]], combined[idx[n:]])
                    .cpu()
                    .detach()
                    .numpy()
                )
                H1_stats.append(
                    test_func(
                        self.subsample(self.X, sample_size).to(self.config.device),
                        self.subsample(self.Y, sample_size).to(self.config.device),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                X_0 = self.subsample(self.X, sample_size).to(config.device)
                H2_stats.append(
                    test_func(X, X_0).cpu().detach().numpy()
                )
            Q_a = np.quantile(np.array(H0_stats), q=0.95)
            Q_b = np.quantile(np.array(H1_stats), q=0.05)

            # print(statistics)
            # print(np.array(statistics))
            power = 1 - (Q_a > np.array(H1_stats)).sum() / num_perm
            type1_error = (Q_b < np.array(H0_stats)).sum() / num_perm

        return power, type1_error, H0_stats, H1_stats, H2_stats

    def run_HT(
        self, num_run, train_X, train_Y, sample_size=200, num_permutations=500, tag=None, lie_group='upper'
    ):
        model = []
        power = []
        type1_error = []
        tags = []

        train_X_dl = DataLoader(train_X.to(self.config.device), 128, shuffle=True)
        train_Y_dl = DataLoader(train_Y.to(self.config.device), 128, shuffle=True)
        trained_char_func = optimize_CF(
            train_X_dl,
            train_Y_dl,
            iterations=2000,
            device=self.config.device,
            M_num_samples=10,
            hidden_size=10,
            input_size=train_X.shape[-1],
            lie_group=lie_group
        )

        for j in tqdm(range(num_run)):
            initial_char_func = char_func_path(
                num_samples=10,
                hidden_size=10,
                input_size=train_X.shape[-1],
                add_time=True,
                init_range=1,
                lie_group=lie_group
            ).to(self.config.device)

            untrained_power, untrained_t1error, H0_stats, H1_stats, H2_stats = self.permutation_test(
                partial(initial_char_func.distance_measure, Lambda=0),
                num_permutations,
                sample_size,
            )
            model.append("PCF"), power.append(untrained_power)
            type1_error.append(untrained_t1error), tags.append(tag)

            # self.hist(H0_stats, H1_stats, H2_stats, lie_group=lie_group, sample_size=sample_size, iter = j)

            trained_power, trained_t1error, H0_stats, H1_stats, H2_stats = self.permutation_test(
                partial(trained_char_func.distance_measure, Lambda=0),
                num_permutations,
                sample_size,
            )
            model.append("Optimized PCF"), power.append(trained_power), tags.append(tag)
            type1_error.append(trained_t1error)
            # self.hist(H0_stats, H1_stats, H2_stats, lie_group=lie_group, sample_size=sample_size, iter = j)

            CF_loss = CFLossFunc()
            CF_power, CF_t1error, H0_stats, H1_stats, H2_stats = self.permutation_test(
                CF_loss, num_permutations, sample_size
            )
            model.append("CF"), power.append(CF_power), tags.append(
                tag
            ), type1_error.append(CF_t1error)

            # self.hist(H0_stats, H1_stats, H2_stats, lie_group=lie_group, sample_size=sample_size, iter=j)
        return pd.DataFrame(
            {"model": model, "power": power, "type1 error": type1_error, "tag": tags}
        )

    def subsample(self, data, sample_size):
        idx = torch.randint(low=0, high=data.shape[0], size=[sample_size])
        return data[idx]


    def hist(self, H0_stats, H1_stats, H2_stats, lie_group, sample_size, iter):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].hist(np.array(H0_stats[:]), bins=25, label='H_0_Permuted', edgecolor='#E6E6E6')
        ax[0].hist(np.array(H1_stats[:]), bins=25, label='H_A', edgecolor='#E6E6E6')

        ax[0].legend(loc='upper right', ncol=2, fontsize=22)
        ax[0].set_xlabel('EPCFD(X, Y)^2', labelpad=10)
        ax[0].set_ylabel('Count', labelpad=10)

        ax[1].hist(np.array(H2_stats[:]), bins=25, label='H_0', edgecolor='#E6E6E6')
        ax[1].hist(np.array(H1_stats[:]), bins=25, label='H_A', edgecolor='#E6E6E6')

        ax[1].legend(loc='upper right', ncol=2, fontsize=22)
        ax[1].set_xlabel('EPCFD(X, Y)^2', labelpad=10)
        ax[1].set_ylabel('Count', labelpad=10)

        plt.tight_layout(pad=3.0)
        # plt.savefig('MMD_binary_tex.pdf',bbox_inches='tight')
        plt.savefig('./numerical_results/HT/hist_h_{}_{}_batch_{}_iter_{}.png'.format(h, lie_group, sample_size, iter))
    #########################
    """moments problem"""


if __name__ == "__main__":
    import ml_collections
    import yaml
    import os

    config_dir = "configs/" + "test_metrics.yaml"
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    from src.datasets.fbm_dl import FBM_data

    sns.set()
    h_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # h_list = [0.45]
    df_list = []
    X = FBM_data(10000, dim=3, length=50, h=0.5)
    train_X = FBM_data(5000, dim=3, length=50, h=0.5)
    for h in h_list:
        Y = FBM_data(10000, dim=3, length=50, h=h)
        train_Y = FBM_data(5000, dim=3, length=50, h=h)
        df = Compare_test_metrics(X, Y, config).run_HT(
            num_run=5, train_X=train_X, train_Y=train_Y, tag=h, lie_group='unitary', sample_size=200, num_permutations=100,
        )
        print(df)
        df_list.append(df)
    df = pd.concat(df_list)

    df.to_csv("numerical_results/HT/metric_compare_fbm_unitary_partition.csv")
