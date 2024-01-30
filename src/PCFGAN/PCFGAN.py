import torch
from torch import nn
from tqdm import tqdm

from src.baselines.base import BaseTrainer
from src.utils import AddTime
from src.PCFGAN.nn import development_layer, development_layer_v2
from torch.nn.functional import one_hot
import torch.optim.swa_utils as swa_utils
import matplotlib.pyplot as plt
from os import path as pt
from src.utils import to_numpy, track_gradient_norms, track_norm
from functools import partial
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class char_func_path(nn.Module):
    def __init__(
        self,
        num_samples,
        hidden_size,
        input_size,
        add_time: bool,
        lie_group: str = 'unitary',
        partition_size: int = 0,
        init_range: float = 1,
    ):
        """
        Class for computing path charateristic function.

        Args:
            num_samples (int): Number of samples.
            hidden_size (int): Hidden size.
            input_size (int): Input size.
            add_time (bool): Whether to add time dimension to the input.
            init_range (float, optional): Range for weight initialization. Defaults to 1.
        """
        super(char_func_path, self).__init__()
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.partition_size = partition_size
        if add_time:
            self.input_size = input_size + 1
        else:
            self.input_size = input_size + 0

        self.lie_group = lie_group
        # self.unitary_development = development_layer(
        #     input_size=self.input_size,
        #     hidden_size=self.hidden_size,
        #     lie_group=self.lie_group,
        #     channels=self.num_samples,
        #     include_inital=True,
        #     return_sequence=False,
        #     init_range=init_range,
        # )
        self.unitary_development = development_layer_v2(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            lie_group=self.lie_group,
            channels=self.num_samples,
            include_inital=True,
            partition_size=self.partition_size,
            init_range=init_range,
        )
        for param in self.unitary_development.parameters():
            param.requires_grad = True
        self.add_time = add_time

    def reset_parameters(self):
        pass

    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor, keep_time_dim=False):
        """
        Hilbert-Schmidt norm computation.

        Args:
            X (torch.Tensor): Complex-valued tensor of shape (C, m, m).
            Y (torch.Tensor): Tensor of the same shape as X.

        Returns:
            torch.float: Hilbert-Schmidt norm of X and Y.
        """
        assert len(X.shape) == len(Y.shape), "The dimension of X and Y must agree."
        # print(X.shape)
        if len(X.shape) == 4:
            # print(keep_time_dim)
            D = torch.einsum("bcij,bcjk->bcik", X, torch.conj(Y).permute(0, 1, 3, 2))
            # return (torch.einsum("bcii->bc", D)).mean().real
            if keep_time_dim:
                # print((torch.einsum("bcii->bc", D)))
                return (torch.einsum("bcii->bc", D)).mean(-1).real
            else:
                return (torch.einsum("bcii->bc", D)).mean().real
        elif len(X.shape) == 3:
            D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
            return (torch.einsum("bii->b", D)).mean().real
        else:
            raise ValueError("The dimension of X must be either 3 or 4.")

    def distance_measure(
        self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1, keep_time_dim = False
    ) -> torch.float:
        """
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """

        if self.partition_size:
            return self.dyadic_distance_measure(X1, X2, Lambda, keep_time_dim)
        # print(X1.shape)
        if self.add_time:
            X1 = AddTime(X1)
            X2 = AddTime(X2)
        else:
            pass
        # print(X1.shape)
        dev1, dev2 = self.unitary_development(X1), self.unitary_development(X2)
        N, T, d = X1.shape

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(0), dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_CF_1 = self.unitary_development(initial_incre_X1).mean(0)
            initial_CF_2 = self.unitary_development(initial_incre_X2).mean(0)
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + Lambda * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2, keep_time_dim=keep_time_dim
            )
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim)


    def dyadic_distance_measure(
        self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1, keep_time_dim = False
    ) -> torch.float:
        """
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """
        # print(X1.shape)
        if self.add_time:
            X1 = AddTime(X1)
            X2 = AddTime(X2)
        else:
            pass
        dev1, dyadic_dev1 = self.unitary_development(X1)
        dev2, dyadic_dev2 = self.unitary_development(X2)
        N, T, d = X1.shape
        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(0), dev2.mean(0)
        dyadic_CF1, dyadic_CF2 = dyadic_dev1.mean(0), dyadic_dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_dev_1, _ = self.unitary_development(initial_incre_X1)
            initial_dev_2, _ = self.unitary_development(initial_incre_X2)

            initial_CF_1 = initial_dev_1.mean(0)
            initial_CF_2 = initial_dev_2.mean(0)

            # initial_dyadic_CF_1, initial_dyadic_CF_2 = initial_dyadic_dev_1.mean(0), initial_dyadic_dev_2.mean(0)

            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + \
                Lambda * self.HS_norm(initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2, keep_time_dim=keep_time_dim) + \
                self.HS_norm(dyadic_CF1 - dyadic_CF2, dyadic_CF1 - dyadic_CF2, keep_time_dim=keep_time_dim)
                # Lambda * self.HS_norm(initial_dyadic_CF_1 - initial_dyadic_CF_2, initial_dyadic_CF_1 - initial_dyadic_CF_2, keep_time_dim=keep_time_dim)
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + self.HS_norm(dyadic_CF1 - dyadic_CF2, dyadic_CF1 - dyadic_CF2, keep_time_dim=keep_time_dim)

class PCFGANTrainer(BaseTrainer):
    def __init__(self, G, train_dl, config, **kwargs):
        """
        Trainer class for the basic PCF-GAN, without time serier embedding module.

        Args:
            G (torch.nn.Module): PCFG generator model.
            train_dl (torch.utils.data.DataLoader): Training data loader.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments for the base trainer class.
        """
        super(PCFGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9), weight_decay=0
            ),
            **kwargs
        )
        print("Using PCFGANTrainer, with lie group {} and dyadic partition size {}".format(config.lie_group,
                                                                                            config.partition_size))

        self.config = config
        self.add_time = config.add_time
        self.train_dl = train_dl
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.G_steps_per_D_step = config.G_steps_per_D_step
        char_input_dim = self.config.input_dim
        self.char_func = char_func_path(
            num_samples=config.M_num_samples,
            hidden_size=config.M_hidden_dim,
            input_size=char_input_dim,
            add_time=self.add_time,
            lie_group=config.lie_group,
            partition_size=config.partition_size
        )
        self.D = self.char_func
        self.char_optimizer = torch.optim.Adam(
            self.char_func.parameters(), lr=config.lr_M
        )
        self.averaged_G = swa_utils.AveragedModel(G)
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=config.gamma
        )
        self.M_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.char_optimizer, gamma=config.gamma
        )

    def fit(self, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """

        self.G.to(device)
        self.char_func.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        """
        Performs one training step.

        Args:
            device: Device to perform training on.
            step (int): Current training step.
        """
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            with torch.no_grad():
                x_real_batch = next(iter(self.train_dl))[0].to(device)
                x_fake = self.G(
                    batch_size=self.batch_size,
                    n_lags=self.config.n_lags,
                    device=device,
                )

            D_loss = self.D_trainstep(x_fake, x_real_batch)
            if i == 0:
                self.losses_history["D_loss"].append(D_loss)

        for i in range(self.G_steps_per_D_step):
            G_loss = self.G_trainstep(x_real_batch, device, step, i)
            self.losses_history["G_loss"].append(G_loss)
        torch.cuda.empty_cache()
        # G_loss = self.G_trainstep(x_real_batch, device, step)
        if step % 500 == 0:
            self.G_lr_scheduler.step()
            for param_group in self.G_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

    def G_trainstep(self, x_real, device, step, i=0):
        """
        Performs one training step for the generator.

        Args:
            x_real: Real samples for training.
            device: Device to perform training on.
            step (int): Current training step.

        Returns:
            float: Generator loss value.
        """
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.char_func.train()
        x_fake = self.G(
            batch_size=self.batch_size,
            n_lags=self.config.n_lags,
            device=device,
        )
        # if self.loss == "both":
        #     G_loss = self.char_func.distance_measure(x_real, x_fake, Lambda=0.1)

        G_loss = self.char_func.distance_measure(x_real, x_fake, keep_time_dim=True)  # (T)
        # print(G_loss.shape)
        self.losses_history['G_loss_dyadic'].append(G_loss)
        G_loss = G_loss.mean()
        G_loss.backward()
        self.losses_history["G_loss"].append(G_loss.item())

        if i == 0:
            grad_norm_G = track_gradient_norms(self.G)
            grad_norm_M1 = track_gradient_norms(self.char_func)
            self.losses_history['grad_norm_G'].append(grad_norm_G)
            self.losses_history['grad_norm_M'].append(grad_norm_M1)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.G_optimizer.step()
        toggle_grad(self.G, False)
        if step % self.config.evaluate_every == 0 and i==0:
            self.evaluate(x_fake, x_real, step, self.config)
            # print(torch.stack(self.losses_history['G_loss_dyadic']).shape)
            plt.plot(to_numpy(torch.stack(self.losses_history['G_loss_dyadic'])))
            plt.savefig(
                pt.join(self.config.exp_dir, "G_loss_dyadic_" + str(step) + ".png")
            )
            plt.close()
            self.plot_losses(loss_item="G_loss", step=step)
            self.plot_losses(loss_item="grad_norm_G", step=step)
            self.plot_losses(loss_item="grad_norm_M", step=step)
            self.plot_losses(loss_item="SigMMD", step=step)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        """
        Performs one training step for the discriminator.

        Args:
            x_fake: Fake samples generated by the generator.
            x_real: Real samples for training.

        Returns:
            float: Discriminator loss value.
        """
        toggle_grad(self.char_func, True)
        self.char_func.train()
        self.char_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_loss = -self.char_func.distance_measure(x_real, x_fake, Lambda=0.1)

        d_loss.backward()

        # Step discriminator params
        self.char_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.char_func, False)

        return d_loss.item()

    def plot_losses(self, loss_item: str, step: int = 0):
        plt.plot(self.losses_history[loss_item])
        plt.savefig(
            pt.join(self.config.exp_dir, loss_item + "_" + str(step) + ".png")
        )
        plt.close()

class RPCFGANTrainer(BaseTrainer):
    def __init__(self, G, D, train_dl, config, **kwargs):
        """
        Trainer class for PCFGAN with time series embedding,
            which provide extrac time series reconstruction functionality.

        Args:
            G (torch.nn.Module): RPCFG generator model.
            D (torch.nn.Module): RPCFG discriminator model (character function).
            train_dl (torch.utils.data.DataLoader): Training data loader.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments for the base trainer class.
        """
        super(RPCFGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)
            ),
            **kwargs
        )
        print("Using RPCFGANTrainer, with lie group {} and dyadic partition size {}".format(config.lie_group, config.partition_size))
        self.config = config
        self.add_time = config.add_time
        self.train_dl = train_dl
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.G_steps_per_D_step = config.G_steps_per_D_step
        self.D = D
        char_input_dim = self.config.D_out_dim

        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=config.lr_D, betas=(0, 0.9)
        )
        self.char_func = char_func_path(
            num_samples=config.M_num_samples,
            hidden_size=config.M_hidden_dim,
            input_size=char_input_dim,
            add_time=self.add_time,
            init_range=config.init_range,
            lie_group=config.lie_group,
            partition_size=config.partition_size
        )
        self.char_func1 = char_func_path(
            num_samples=config.M_num_samples,
            hidden_size=config.M_hidden_dim,
            input_size=char_input_dim,
            add_time=self.add_time,
            init_range=config.init_range,
            lie_group=config.lie_group,
            partition_size=config.partition_size
        )

        self.char_optimizer = torch.optim.Adam(
            self.char_func.parameters(), betas=(0, 0.9), lr=config.lr_M
        )
        self.char_optimizer1 = torch.optim.Adam(
            self.char_func1.parameters(), betas=(0, 0.9), lr=config.lr_M
        )

        self.averaged_G = swa_utils.AveragedModel(G)
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=config.gamma
        )
        self.D_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.D_optimizer, gamma=config.gamma
        )
        self.M_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.char_optimizer, gamma=config.gamma
        )
        self.M_lr_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
            self.char_optimizer1, gamma=config.gamma
        )
        self.Lambda1 = self.config.Lambda1
        self.Lambda2 = self.config.Lambda2
        if self.config.BM:
            self.noise_scale = self.config.noise_scale
        else:
            self.noise_scale = 0.3

    def fit(self, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """
        self.G.to(device)
        self.D.to(device)
        self.char_func.to(device)
        self.char_func1.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        """
        Performs one training step.

        Args:
            device: Device to perform training on.
            step (int): Current training step.
        """
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            with torch.no_grad():
                z = (
                    self.noise_scale
                    * torch.randn(
                        self.config.batch_size,
                        self.config.n_lags,
                        self.config.G_input_dim,
                    )
                ).to(device)
                if self.config.BM:
                    z = z.cumsum(1)
                else:
                    pass
                x_real_batch = next(iter(self.train_dl))[0].to(device)
                x_fake = self.G(
                    batch_size=self.batch_size,
                    n_lags=self.config.n_lags,
                    z=z,
                    device=device,
                )

            self.M_trainstep(x_fake, x_real_batch, z)
            D_loss, enc_loss, reg_loss = self.D_trainstep(
                x_fake, x_real_batch, z, self.Lambda1, self.Lambda2
            )
            if i == 0:
                self.losses_history["D_loss"].append(D_loss)
                self.losses_history["recovery_loss"].append(enc_loss)
                self.losses_history["regularzation_loss"].append(reg_loss)

        for i in range(self.G_steps_per_D_step):
            G_loss = self.G_trainstep(x_real_batch, device, step, i)
            self.losses_history["G_loss"].append(G_loss)
        torch.cuda.empty_cache()
        if step % 500 == 0:
            self.D_lr_scheduler.step()
            self.G_lr_scheduler.step()
            for param_group in self.D_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

        if step % self.config.evaluate_every == 0:
            latent_x_fake = self.D(x_fake)
            latent_x_real = self.D(x_real_batch)
            x_real_dim = latent_x_fake.shape[-1]
            for j in range(x_real_dim):
                plt.plot(to_numpy(latent_x_fake[:100, :, j]).T, "C%s" % j, alpha=0.1)
            plt.savefig(
                pt.join(self.config.exp_dir, "latent_x_fake_" + str(step) + ".png")
            )
            plt.close()
            for j in range(x_real_dim):
                plt.plot(to_numpy(latent_x_real[:100, :, j]).T, "C%s" % j, alpha=0.1)
            plt.savefig(
                pt.join(self.config.exp_dir, "latent_x_real_" + str(step) + ".png")
            )
            plt.close()

    def G_trainstep(self, x_real, device, step, i=0):
        """
        Performs one training step for the generator.

        Args:
            x_real: Real samples for training.
            device: Device to perform training on.
            step (int): Current training step.

        Returns:
            float: Generator loss value.
        """
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.D.train()
        self.char_func.train()
        x_fake = self.G(
            batch_size=self.batch_size, n_lags=self.config.n_lags, device=device
        )
        G_loss = self.char_func.distance_measure(self.D(x_real), self.D(x_fake), keep_time_dim=False) # (T)
        # print(G_loss.shape)
        # self.losses_history['G_loss_dyadic'].append(G_loss)
        # G_loss = G_loss.mean()
        G_loss.backward()
        if i == 0:
            norm_G = track_norm(self.G)
            norm_D = track_norm(self.D)
            norm_M1 = track_norm(self.char_func)
            norm_M2 = track_norm(self.char_func1)
            grad_norm_G = track_gradient_norms(self.G)
            grad_norm_D = track_gradient_norms(self.D)
            grad_norm_M1 = track_gradient_norms(self.char_func)
            grad_norm_M2 = track_gradient_norms(self.char_func1)
            self.losses_history['norm_G'].append(norm_G)
            self.losses_history['norm_D'].append(norm_D)
            self.losses_history['norm_M'].append(norm_M1)
            self.losses_history['norm_M_regularization'].append(norm_M2)
            self.losses_history['grad_norm_G'].append(grad_norm_G)
            self.losses_history['grad_norm_D'].append(grad_norm_D)
            self.losses_history['grad_norm_M'].append(grad_norm_M1)
            self.losses_history['grad_norm_M_regularization'].append(grad_norm_M2)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.G_optimizer.step()
        toggle_grad(self.G, False)
        if step % self.config.evaluate_every == 0 and i==0:
            print('Loss D: ', self.losses_history['grad_norm_D'][-1])
            with torch.no_grad():
                x_fake = self.G(
                    batch_size=self.batch_size, n_lags=self.config.n_lags, device=device
                )
            self.evaluate(x_fake, x_real, step, self.config)
            # print(torch.stack(self.losses_history['G_loss_dyadic']).shape)
            # plt.plot(to_numpy(torch.stack(self.losses_history['G_loss_dyadic'])))
            # plt.savefig(
            #     pt.join(self.config.exp_dir, "G_loss_dyadic_" + str(step) + ".png")
            # )
            # plt.close()
            self.plot_losses(loss_item="G_loss", step=step)
            self.plot_losses(loss_item="recovery_loss", step=step)
            self.plot_losses(loss_item="regularzation_loss", step=step)
            self.plot_losses(loss_item="grad_norm_G", step=step)
            self.plot_losses(loss_item="grad_norm_D", step=step)
            self.plot_losses(loss_item="grad_norm_M", step=step)
            self.plot_losses(loss_item="grad_norm_M_regularization", step=step)
            self.plot_losses(loss_item="norm_G", step=step)
            self.plot_losses(loss_item="norm_D", step=step)
            self.plot_losses(loss_item="norm_M", step=step)
            self.plot_losses(loss_item="norm_M_regularization", step=step)
            self.plot_losses(loss_item="SigMMD", step=step)
            self.plot_losses(loss_item="marginal_distribution", step=step)

        return G_loss.item()

    def M_trainstep(self, x_fake, x_real, z):
        """
        Performs one training step for the character function.

        Args:
            x_fake: Fake samples generated by the generator.
            x_real: Real samples for training.
            z: Latent noise used for generating fake samples.
        """
        toggle_grad(self.char_func, True)
        self.char_func.train()
        self.D.train()
        self.char_optimizer.zero_grad()
        char_loss = -self.char_func.distance_measure(self.D(x_real), self.D(x_fake))
        char_loss.backward()
        self.char_optimizer.step()
        toggle_grad(self.char_func, False)

        toggle_grad(self.char_func1, True)
        self.char_func1.train()
        self.char_optimizer1.zero_grad()
        char_loss1 = -self.char_func1.distance_measure(self.D(x_real), z)
        char_loss1.backward()
        self.char_optimizer1.step()
        toggle_grad(self.char_func1, False)

    def D_trainstep(self, x_fake, x_real, z, Lambda1, Lambda2):
        """
        Performs one training step for the discriminator.

        Args:
            x_fake: Fake samples generated by the generator.
            x_real: Real samples for training.
            z: Latent noise used for generating fake samples.
            Lambda1: Weight for the reconstruction loss.
            Lambda2: Weight for the regularization loss.

        Returns:
            Tuple[float, float, float]: Discriminator loss, reconstruction loss, and regularization loss.
        """
        x_real.requires_grad_()
        self.D.train()

        self.char_func.train()
        self.char_func1.train()
        toggle_grad(self.D, True)
        self.D_optimizer.zero_grad()
        rec_loss = nn.MSELoss()(self.D(x_fake), z)
        reg_loss = self.char_func1.distance_measure(self.D(x_real), z)
        g_loss = -self.char_func.distance_measure(self.D(x_real), self.D(x_fake))

        d_loss = g_loss + Lambda1 * rec_loss + Lambda2 * reg_loss
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.config.grad_clip)
        self.D_optimizer.step()
        toggle_grad(self.D, False)

        return g_loss.item(), rec_loss.item(), reg_loss.item()

    def plot_losses(self, loss_item: str, step: int = 0):
        plt.plot(self.losses_history[loss_item])
        plt.savefig(
            pt.join(self.config.exp_dir, loss_item + "_" + str(step) + ".png")
        )
        plt.close()