####################################################################################################
#                                         frameworkNN.py                                           #
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

# own
from frameworks.framework import Framework
from models.nnModels import *
from simulation.dataModules import SynthDataModule, ChallengeDataModule
from simulation.sigModels import VoigtModel
from simulation.simulation import extract_prior_params
from utils.processing import processBasis, processSpectra


#**************************************************************************************************#
#                                          Class Framework                                         #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test data-driven models.                            #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkNN(Framework, pl.LightningModule):
    def __init__(self, path2basis, basisFmt='', specType='synth', dataType='none', ppmlim=None,
                 val_size=1000, batch=16, lr=1e-3, reg_l1=0.0, reg_l2=0.0, loss='mse', arch='mlp',
                 **kwargs):
        Framework.__init__(self, path2basis, basisFmt, specType, dataType, ppmlim)
        pl.LightningModule.__init__(self)
        # self.save_hyperparameters(ignore=['net'])

        # training parameters
        self.valSize = val_size   # validation set size
        self.batchSize = batch
        self.lr = lr
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.lossType = loss

        # network parameters
        self.kwargs = kwargs

        self.basis = processBasis(self.basisObj.fids)
        self.sigModel = VoigtModel(basis=self.basis, first=self.first, last=self.last,
                                   t=self.basisObj.t, f=self.basisObj.f)

        self.dataSynth = SynthDataModule(basis_dir=path2basis,
                                         nums_test=-1,
                                         sigModel=self.sigModel,
                                         params=self.ps,
                                         concs=self.concs,
                                         basisFmt=basisFmt,)
        self.net = self.setModel(modelType=arch).to(self.device) if arch is not None else None

        # compute priors for parameters (including concentrations)
        self.p_min, self.p_max = extract_prior_params(self.dataSynth.basisObj,
                                                      self.dataSynth.params,
                                                      self.dataSynth.concs,
                                                      mode='unif')
        self.p_min, self.p_max = torch.tensor(self.p_min), torch.tensor(self.p_max)


    #***********************#
    #   configure network   #
    #***********************#
    def setModel(self, modelType):
        input_size = (2, (self.last - self.first) // self.skip)
        if modelType.lower() == 'mlp':
            model = MLP(input_size , self.basis.shape[1], **self.kwargs)
        elif modelType.lower() == 'cnn':
            model = CNN(input_size, self.basis.shape[1], **self.kwargs)
        else: raise ValueError(f'Unknown model: {modelType}')
        return model


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x_ref=None, frac=None, x0=None):
        assert x_ref is None, 'Water referencing not supported!'
        assert frac is None, 'Tissue correction not supported!'
        assert x0 is None, 'Initial parameters not supported!'

        xs = x[:, :, self.first:self.last:self.skip]
        theta = self.net(xs.float())
        return theta


    #**********************#
    #   loss calculation   #
    #**********************#
    def loss(self, x, y, t, theta, type='mae_conc'):
        if type.lower() == 'mae_conc':
            t = torch.clamp(t[:, :self.basis.shape[1]], min=0)   # only positive concentrations
            theta = torch.clamp(theta[:, :self.basis.shape[1]], min=0)

            # exclude macromolecules
            idx = [i for i, name in enumerate(self.basisObj.names)
                   if 'mm' in name.lower() or 'mac' in name.lower()]
            t[:, idx] = 0
            theta[:, idx] = 0

            return torch.nn.L1Loss(reduction='none')(t, theta)

        elif type.lower() == 'mae_conc_opt':
            t = torch.clamp(t[:, :self.basis.shape[1]], min=0)
            theta = torch.clamp(theta[:, :self.basis.shape[1]], min=0)
            w = self.optimalReference(t, theta, type='scale')
            return torch.nn.L1Loss(reduction='none')(t, w * theta)

        elif type.lower() == 'mae_all':
            return torch.nn.L1Loss(reduction='none')(theta, t)

        elif type.lower() == 'mae_all_scale':
            # scale the output with the priors
            self.p_min, self.p_max = self.p_min.to(t.device), self.p_max.to(t.device)
            theta = (theta - self.p_min) / (self.p_max - self.p_min)
            t = (t - self.p_min) / (self.p_max - self.p_min)
            return torch.nn.L1Loss(reduction='none')(theta, t)

        elif type.lower() == 'mse_all':
            return torch.nn.MSELoss(reduction='none')(theta, t)

        elif type.lower() == 'mse_specs':
            spec = self.sigModel.forward(theta)
            spec = processSpectra(spec, self.basis)

            err = torch.nn.MSELoss(reduction='none')(spec[:, 0, self.first:self.last],
                                                     x[:, 0, self.first:self.last])
            err += torch.nn.MSELoss(reduction='none')(spec[:, 1, self.first:self.last],
                                                      x[:, 1, self.first:self.last])
            return err

        elif type.lower() == 'mse_specs_norm':
            spec = self.sigModel.forward(theta)
            spec = processSpectra(spec, self.basis)

            # normalize
            x = x / torch.abs(x[:, 0] + 1j * x[:, 1]).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            spec = spec / torch.abs(spec[:, 0] + 1j * spec[:, 1]).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

            err = torch.nn.MSELoss(reduction='none')(spec[:, 0, self.first:self.last],
                                                     x[:, 0, self.first:self.last])
            err += torch.nn.MSELoss(reduction='none')(spec[:, 1, self.first:self.last],
                                                      x[:, 1, self.first:self.last])
            return err

        elif type.lower() == 'cos_specs':
            spec = self.sigModel.forward(theta)
            spec = processSpectra(spec, self.basis)

            err = torch.nn.CosineSimilarity(dim=-1)(spec[:, 0, self.first:self.last],
                                                    x[:, 0, self.first:self.last])
            err += torch.nn.CosineSimilarity(dim=-1)(spec[:, 1, self.first:self.last],
                                                     x[:, 1, self.first:self.last])
            return err

        elif type.lower() == 'err_func':
            return self.sigModel.err_func(theta, x[:, 0] + 1j * x[:, 1], self.first, self.last)

        else:
            raise ValueError(f'Unknown loss type: {type}')


    #*******************************#
    #   defines the training loop   #
    #*******************************#
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y, t = batch
        y_hat = self.forward(x)

        loss = self.loss(x, y, t, y_hat, type=self.lossType)
        self.log("train_loss", loss.mean().item(), prog_bar=True)

        thetaLoss = self.loss(x, y, t, y_hat, type='mae_conc')
        self.log("train_conc_loss", thetaLoss.mean().item(), prog_bar=True)

        # regularization
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss = loss + self.reg_l1 * l1_norm + self.reg_l2 * l2_norm

        return loss.mean()


    #*********************************#
    #   defines the validation loop   #
    #*********************************#
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, t = batch
        y_hat = self.forward(x)
        loss = self.loss(x, y, t, y_hat, type=self.lossType)
        self.log("val_loss", loss.mean().item(), prog_bar=True)
        thetaLoss = self.loss(x, y, t, y_hat, type='mae_conc')
        self.log("val_conc_loss", thetaLoss.mean().item(), prog_bar=True)


    #***************************#
    #   defines the test loop   #
    #***************************#
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, t = batch
        y_hat = self.forward(x)
        loss = self.loss(x, y, t, y_hat, type=self.lossType)
        self.log("test_loss", loss.mean().item())


    #**************************#
    #   training data loader   #
    #**************************#
    def train_dataloader(self):
        while True:  # ad-hoc simulation
            yield self.dataSynth.get_batch(self.batchSize)


    #****************************#
    #   validation data loader   #
    #****************************#
    def val_dataloader(self):
        data = []
        for _ in range(self.valSize):
            x, y, t = self.dataSynth.get_batch(1)
            data.append([x[0], y[0], t[0]])
        return DataLoader(data, batch_size=self.batchSize)


    #*************************#
    #   testing data loader   #
    #*************************#
    def test_dataloader(self):
        return ChallengeDataModule(basis_dir=self.basis_dir).test_dataloader()


    #*****************************#
    #   configure the optimizer   #
    #*****************************#
    def configure_optimizers(self):        
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                               mode='min', factor=0.5, patience=10000,
        #                               cooldown=0, min_lr=0, eps=1e-8)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        return optimizer
