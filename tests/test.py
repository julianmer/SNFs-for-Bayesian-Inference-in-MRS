####################################################################################################
#                                             test.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/12/24                                                                                #
#                                                                                                  #
# Purpose: Testing of models with different data.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import shutup; shutup.please()   # shut up warnings
import torch

from tqdm import tqdm

# own
from frameworks.framework import Framework
from utils.gpu_config import set_gpu_usage
from utils.structures import Map



#**************************************************************************************************#
#                                            Class Test                                            #
#**************************************************************************************************#
#                                                                                                  #
# The main class for testing the models.                                                           #
#                                                                                                  #
#**************************************************************************************************#
class Test():

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self, config):
            self.config = Map(config)  # make config mappable
            self.system = self.getSysModel(self.config)
            self.model = self.getModel(self.config)  # get the model
            self.dataloader = self.getDataLoader(self.config)  # get the data loader


        #****************#
        #   model init   #
        #****************#
        def getModel(self, config):
            config = Map(config)  # make config mappable
            if config.model == 'lcm':
                if config.method.lower() == 'newton' or config.method.lower() == 'mh':
                    from frameworks.frameworkFSL import FrameworkFSL
                    model = FrameworkFSL(**config)
                elif config.method.lower() == 'lcmodel':
                    from frameworks.frameworkLCM import FrameworkLCModel
                    model = FrameworkLCModel(**config)
                else:
                    raise ValueError('Method not recognized.')
            else:
                if config.model == 'nn':
                    from frameworks.frameworkNN import FrameworkNN
                    model = FrameworkNN(**config)
                elif config.model.lower() == 'snf':
                    from frameworks.frameworkSNF import FrameworkSNF
                    model = FrameworkSNF(**config)
                else:
                    raise ValueError('Model %s is not recognized' % config.model)

                if config.load_model:
                    gpu = torch.device(set_gpu_usage() if torch.cuda.is_available() else 'cpu')
                    model = type(model).load_from_checkpoint(**config, map_location=gpu)

            return model


        #***********************#
        #   system model init   #
        #***********************#
        def getSysModel(self, config):
            return Framework(path2basis=config.path2basis, basisFmt=config.basisFmt,
                             specType=config.specType, dataType=config.dataType,
                             ppmlim=config.ppmlim)


        #**********************#
        #   data loader init   #
        #**********************#
        def getDataLoader(self, config, sigModel=None, params=None, concs=None):
            if sigModel is None: sigModel = self.system.sigModel
            if params is None: params = self.system.ps
            if concs is None: concs = self.system.concs

            if config.dataType[:3] == 'cha':
                from simulation.dataModules import ChallengeDataModule
                dataloader = ChallengeDataModule(basis_dir=config.path2basis,
                                                 nums_cha=config.test_size)

            else:
                from simulation.dataModules import SynthDataModule
                dataloader = SynthDataModule(basis_dir=config.path2basis,
                                             nums_test=config.test_size,
                                             sigModel=sigModel,
                                             params=params,
                                             concs=concs,
                                             basisFmt=config.basisFmt,
                                             specType=config.specType)
            return dataloader


        #***********************#
        #   run model on data   #
        #***********************#
        def run(self, model=None, data=None):
            if model is None: model = self.model
            if data is None: data = self.dataloader.test_dataloader()

            truths, preds = [], []
            specs, specs_sep = [], []
            with torch.no_grad():
                for x, y, t in tqdm(data):
                    specs.append(x)
                    specs_sep.append(y)
                    t_hat = model.forward(x.to(model.device) if hasattr(model, 'device') else x)
                    truths.append(t)
                    if isinstance(t_hat, tuple): t_hat = t_hat[0]   # only concentrations
                    if not isinstance(t_hat, torch.Tensor): t_hat = torch.Tensor(t_hat)
                    if not hasattr(model, 'basisObj'):
                        t_hat = torch.stack([t_hat[:, model.basisFSL.names.index(m)]   # sort
                                             for m in self.system.basisObj.names], dim=1)
                    preds.append(t_hat)

            return torch.cat(truths), torch.cat(preds), torch.cat(specs), torch.cat(specs_sep)



#*************#
#   testing   #
#*************#
if __name__ == '__main__':

    # initialize the configuration
    config = {
        # path to a trained model
        'checkpoint_path': '',

        # path to basis set
        'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',

        # path to data
        'path2data': '',   # in-vivo data
        'control': '../Data/Other/Synth.control',   # control file for LCModel

        # model
        'model': 'lcm',  # 'nn', 'lcm', 'snf', ...
        'loss': 'mse',  # 'mae', 'mse', ...

        # setup
        'specType': 'auto',  # 'auto', 'synth', 'invivo', 'fMRSinPain', 'all'
        'ppmlim': (0.5, 4.5),  # ppm limits for the spectra (used if specType='auto')
        'basisFmt': '7tslaser',  # '', 'cha_philips', 'biggaba', '7tslaser', ...
        'dataType': 'aumc2',  # 'aumc', 'aumc2', 'aumc_mod', 'custom', ...
        'test_size': 1000,  # number of test samples

        # for nn model
        'arch': 'mlp',  # 'mlp', 'cnn' ...

        # for lcm model
        'method': 'LCModel',  # 'Newton', 'MH', 'LCModel'
        'bandwidth': 3000,  # bandwidth of the spectra
        'sample_points': 1024,  # number of sample points
        'save_path': '', #'./testLCM/',  # path to save the results (empty for no saving)

        # mode
        'load_model': False,  # load model from path2trained
        'skip_train': True,  # skip the training procedure

        # visual settings
        'run': True,  # run the inference (will try to load results if False)
        'save': True,  # save the plots
        'error': 'msmae',  # 'mae', 'mse', 'mape', ...
    }

    # run a quick test
    test = Test(config)

    if test.config.dataType != 'invivo':
        truths, preds, _, _ = test.run()
        err = test.system.concsLoss(truths, preds, type=test.config.error)
        print(f'{test.config.model} {test.config.method if test.config.model == "lcm" else ""} '
              f'error: {err.mean().item()} ± {err.std().item() / np.sqrt(len(err))}')

        ccc = test.system.concsLoss(truths, preds, type='msccc')
        print(f'{test.config.model} {test.config.method if test.config.model == "lcm" else ""} '
              f'CCC: {ccc.mean().item()} ± {ccc.std().item() / np.sqrt(len(ccc))}')

    else:   # run in-vivo test

        import numpy as np
        from loading.loadData import loadDataAsFSL

        # load and prepare the data
        data = loadDataAsFSL(config['path2data'], fmt='philips')

        x = np.fft.fft(data.FID)
        x = np.stack([x.real, x.imag])
        x = torch.tensor(x).unsqueeze(0).float()

        # run the model
        with torch.no_grad():
            test.model.save_path = test.config.save_path
            t_hat = test.model.forward(x)