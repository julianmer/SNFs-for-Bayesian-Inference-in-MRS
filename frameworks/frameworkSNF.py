####################################################################################################
#                                         frameworkSNF.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Optimization framework consisting of neural models as well as training and              #
#          testing loops.                                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from tqdm import tqdm

# own
from frameworks.frameworkNN import FrameworkNN


#**************************************************************************************************#
#                                          Class Framework                                         #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test data-driven models.                            #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkSNF(FrameworkNN, pl.LightningModule):
    def __init__(self, path2basis, basisFmt='', specType='synth', dataType='none', ppmlim=None,
                 val_size=1000, batch=16, lr=1e-3, reg_l1=0.0, reg_l2=0.0, loss='mse', arch='mlp',
                 flow_args=None, **kwargs):

        # initialize parent classes
        FrameworkNN.__init__(self, path2basis, basisFmt, specType, dataType, ppmlim, val_size,
                             batch, lr, reg_l1, reg_l2, loss, arch, **kwargs)
        pl.LightningModule.__init__(self)
        # self.save_hyperparameters(ignore=['net'])

        # compute scaling from basis
        self.scale = torch.fft.fft(torch.tensor(self.basisObj.fids), dim=0).mean().real

        # flow arguments
        self.args = {
            'flow': 'IAFVAE',
            'input_dim': (2, int(self.last - self.first)),
            'latent_dim': self.basisObj.n_metabs + 11,

            # # encoder and decoder type
            'decoder': 'model_based',
            'sigModel': self.sigModel,  # must be passed with 'model_based' decoder
            'dataSynth': self.dataSynth,

            # loss
            'loss': loss,
            'beta': 1.0,

            # vae architecture
            'arch': arch,
            'dropout': kwargs['dropout'] if 'dropout' in kwargs else 0.0,
            'width': kwargs['width'] if 'width' in kwargs else 512,
            'depth': kwargs['depth'] if 'depth' in kwargs else 3,
            'activation': kwargs['activation'] if 'activation' in kwargs else 'elu',

            # flow arguments
            'num_flows': 8,
            'hidden_dim': 512,
            'num_hidden': 3,
            'num_ortho_vecs': 8,
            'num_householder': 8,

            'device': kwargs['device'] if 'device' in kwargs else 'cpu',
        }

        # update flow arguments
        if flow_args is not None: self.args.update(flow_args)

        self.flow = self.setFlow(self.args).to(self.device)

        # load vae if specified
        if 'path2vae' in kwargs and not (kwargs['path2vae'] is None or kwargs['path2vae'] == ''):
            vae = type(self).load_from_checkpoint(checkpoint_path=kwargs['path2vae'],
                                                  path2basis=path2basis, basisFmt=basisFmt,
                                                  specType=specType, dataType=dataType, ppmlim=ppmlim,
                                                  val_size=val_size, batch=batch, lr=lr, reg_l1=reg_l1,
                                                  reg_l2=reg_l2, loss=loss, arch=arch,
                                                  flow_args={'flow': 'BaseVAE'}, **kwargs)
            self.flow.q_z_nn = vae.flow.q_z_nn
            self.flow.q_z_mean = vae.flow.q_z_mean
            self.flow.q_z_var = vae.flow.q_z_var

            # freeze vae
            for param in self.flow.q_z_nn.parameters(): param.requires_grad = False
            for param in self.flow.q_z_mean.parameters(): param.requires_grad = False
            for param in self.flow.q_z_var.parameters(): param.requires_grad = False


    #***********************#
    #   configure network   #
    #***********************#
    def setFlow(self, args):
        if args['flow'].lower() == 'basevae':
            from models.flowModels import BaseVAE
            return BaseVAE(**args)
        elif args['flow'].lower() == 'iafvae':
            from models.flowModels import IAFVAE
            return IAFVAE(**args)
        elif args['flow'].lower() == 'snfvae':
            from models.flowModels import SylvesterVAE
            return SylvesterVAE(**args)
        else:
            raise ValueError('Invalid flow choice')


    #***********************#
    #   loss for training   #
    #***********************#
    def flow_loss(self, x, x_mean, z_mu, z_var, z0, zk, ldj, type='nll', beta=1.0):
        if type == 'nll':
            # reconstruction loss
            recon_loss = torch.nn.MSELoss(reduction='none')(x_mean, x).mean(dim=(1, 2))

            # approximate posterior q(z0|x) using the Gaussian parameters
            q_z0 = torch.distributions.Normal(z_mu, torch.exp(0.5 * z_var))
            log_q_z0 = q_z0.log_prob(z0).sum(dim=1)

            # standard Normal prior
            prior = torch.distributions.Normal(torch.zeros_like(z_mu), torch.ones_like(z_mu))
            log_p_zk = prior.log_prob(zk).sum(dim=1)

            # effective KL divergence accounting for the flow transform
            kl = log_q_z0 - ldj - log_p_zk
            return recon_loss + beta * kl, recon_loss, kl

        elif type == 'rvdb':   # loss as defined in the RVDB paper
            batch_size = x.size(0)

            # - N E_q0 [ ln p(x|z_k) ]
            rec = (x - x_mean).pow(2).mean(dim=(1, 2)).sum()

            # ln p(z_k)  (not averaged)
            log_p_zk = torch.sum(-0.5 * zk * zk, dim=1)
            # ln q(z_0)  (not averaged)
            log_q_z0 = torch.sum(-0.5 * (z_var + (z0 - z_mu) * (z0 - z_mu) * z_var.exp().reciprocal()), dim=1)
            # N E_q0[ ln q(z_0) - ln p(z_k) ]
            summed_logs = torch.sum(log_q_z0 - log_p_zk)

            # sum over batches
            summed_ldj = torch.sum(ldj)

            # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
            kl = (summed_logs - summed_ldj)
            loss = rec + beta * kl

            loss = loss / float(batch_size)
            rec = rec / float(batch_size)
            kl = kl / float(batch_size)
            return loss, rec, kl

        elif type == 'vae':
            # reconstruction loss (MSE)
            recon_loss = torch.nn.functional.mse_loss(x_mean, x, reduction='mean')

            # KL divergence for q(z0|x) = N(z_mu, exp(z_var))
            # (assumes z_var is log variance)
            kl_per_sample = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1 - z_var, dim=1)
            kl_loss = torch.mean(kl_per_sample)

            # flow correction:
            #    the change-of-variables formula tells us that:
            #       log q(zk) = log q(z0) - ldj.
            flow_corr = torch.mean(ldj)

            # total loss: reconstruction loss + beta * (KL divergence - flow correction)
            total_loss = recon_loss + beta * (kl_loss - flow_corr)
            return total_loss, recon_loss, kl_loss

        else:
            raise ValueError('Unknown loss type: %s' % type)


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x_ref=None, frac=None, x0=None):
        assert x_ref is None, 'Water referencing not supported!'
        assert frac is None, 'Tissue correction not supported!'
        assert x0 is None, 'Initial parameters not supported!'

        # prepare input
        xs = x[..., self.first:self.last]
        xs /= self.scale

        # forward pass
        x_mean, z_mu, z_var, ldj, z0, zk = self.flow(xs.float())
        x_mean /= self.scale

        return xs, x_mean, z_mu, z_var, ldj, z0, zk


    #********************#
    #   forward sample   #
    #********************#
    def forward_sample(self, x, x_ref=None, frac=None, x0=None, n_samples=1):
        assert x_ref is None, 'Water referencing not supported!'
        assert frac is None, 'Tissue correction not supported!'
        assert x0 is None, 'Initial parameters not supported!'

        # prepare input
        xs = x[..., self.first:self.last]
        xs /= self.scale

        # forward pass
        specs = torch.zeros((n_samples, xs.shape[0]) + self.args['input_dim'])
        thetas = torch.zeros((n_samples, xs.shape[0], self.args['latent_dim']))
        for i in tqdm(range(n_samples)):
            x_mean, z_mu, z_var, ldj, z0, zk = self.flow(xs.float())
            x_mean /= self.scale
            specs[i] = x_mean
            thetas[i] = self.flow.apply_prior(zk)

        return xs, specs, thetas


    #*******************************#
    #   defines the training loop   #
    #*******************************#
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y, t = batch
        x, x_mean, z_mu, z_var, ldj, z0, zk = self.forward(x)
        loss, rec, kl = self.flow_loss(x, x_mean, z_mu, z_var, z0, zk, ldj,
                                       self.args['loss'], self.args['beta'])
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_rec", rec.item(), prog_bar=True)
        self.log("train_kl", kl.item(), prog_bar=True)
        concLoss = self.loss(x, y, t, self.flow.apply_prior(zk), type='mae_conc')
        self.log("train_conc_loss", concLoss.mean().item(), prog_bar=True)
        return loss


    #*********************************#
    #   defines the validation loop   #
    #*********************************#
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, t = batch
        x, x_mean, z_mu, z_var, ldj, z0, zk = self.forward(x)
        loss, rec, kl = self.flow_loss(x, x_mean, z_mu, z_var, z0, zk, ldj,
                                       self.args['loss'], self.args['beta'])
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_rec", rec.item(), prog_bar=True)
        self.log("val_kl", kl.item(), prog_bar=True)
        concLoss = self.loss(x, y, t, self.flow.apply_prior(zk), type='mae_conc')
        self.log("val_conc_loss", concLoss.mean().item(), prog_bar=True)


    #***************************#
    #   defines the test loop   #
    #***************************#
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, t = batch
        x, x_mean, z_mu, z_var, ldj, z0, zk = self.forward(x)
        loss, rec, kl = self.flow_loss(x, x_mean, z_mu, z_var, z0, zk, ldj,
                                       self.args['loss'], self.args['beta'])
        self.log("test_loss", loss.item())
        self.log("test_rec", rec.item())
        self.log("test_kl", kl.item())
        concLoss = self.loss(x, y, t, self.flow.apply_prior(zk), type='mae_conc')
        self.log("test_conc", concLoss.mean().item())