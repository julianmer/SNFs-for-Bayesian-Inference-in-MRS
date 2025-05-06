####################################################################################################
#                                            testSNF.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 11/01/25                                                                                #
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

import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutup; shutup.please()   # shut up warnings
import torch

from copy import deepcopy

from matplotlib.patches import Rectangle

from scipy.stats import norm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tqdm import tqdm

# own
from test import Test



#*************#
#   testing   #
#*************#
if __name__ == '__main__':

    # initialize the configuration
    config = {
        # path to a trained model
        # 'checkpoint_path': './wandb/run-20250426_162820-7xq36sk5/files/checkpoints/last.ckpt',   # baseVAE (beta 10), aumc
        # 'checkpoint_path': './wandb/run-20250426_163037-8q1cva67/files/checkpoints/last.ckpt',   # IAFVAE (beta 10), aumc
        # 'checkpoint_path': './wandb/run-20250426_163133-m87vw8pi/files/checkpoints/last.ckpt',  # SNFVAE (beta 10), aumc

        # 'checkpoint_path': './wandb/run-20250426_163231-2dpdy69e/files/checkpoints/last.ckpt',  # baseVAE (beta 10), aumc2
        # 'checkpoint_path': './wandb/run-20250426_163353-rebkiwtt/files/checkpoints/last.ckpt',  # IAFVAE (beta 10), aumc2
        'checkpoint_path': './wandb/run-20250426_163509-qf09zwe2/files/checkpoints/last.ckpt',  # SNFVAE (beta 10), aumc2

        # 'checkpoint_path': './wandb/run-20250426_163618-4ovuk86h/files/checkpoints/last.ckpt',  # baseVAE (beta 10), aumc_mod
        # 'checkpoint_path': './wandb/run-20250426_163744-bwrjbmgj/files/checkpoints/last.ckpt',  # IAFVAE (beta 10), aumc_mod
        # 'checkpoint_path': './wandb/run-20250426_163916-s8jvq2jt/files/checkpoints/last.ckpt',  # SNFVAE (beta 10), aumc_mod

        # 'checkpoint_path': './wandb/run-20250426_164000-ln9yf8lv/files/checkpoints/last.ckpt',  # baseVAE (beta 10), aumc_mod2
        # 'checkpoint_path': './wandb/run-20250426_164103-8pxo2nb5/files/checkpoints/last.ckpt',  # IAFVAE (beta 10), aumc_mod2
        # 'checkpoint_path': './wandb/run-20250426_164140-9fxsv7yr/files/checkpoints/last.ckpt',  # SNFVAE (beta 10), aumc_mod2

        # 'checkpoint_path': './wandb/run-20250426_163231-2dpdy69e/files/checkpoints/epoch=0-step=541184.ckpt',  # baseVAE (beta 10), aumc2
        # 'checkpoint_path': './wandb/run-20250426_163353-rebkiwtt/files/checkpoints/epoch=0-step=342528.ckpt',  # IAFVAE (beta 10), aumc2
        # 'checkpoint_path': './wandb/run-20250426_163509-qf09zwe2/files/checkpoints/epoch=0-step=367872.ckpt',  # SNFVAE (beta 10), aumc2

        # path to basis set
        'path2basis': '../Data/BasisSets/7T_sLASER_OIT_TE34.basis',
        'control': '../Data/Other/Synth.control',   # control file for LCModel

        # model
        'model': 'snf',  # 'nn', 'lcm', 'snf', ...
        'num_samples': 1000,  # number of latent samples
        'ex_idx': list(range(1000)),  # example index

        # for nn models
        'arch': 'mlp',  # 'mlp', 'cnn' ...
        'activation': 'elu',  # 'relu', 'elu', 'tanh', 'sigmoid', 'softplus', ...
        'dropout': 0.0,
        'width': 512,
        'depth': 3,
        'conv_depth': 5,
        'kernel_size': 5,
        'stride': 2,

        # for snf model
        'flow_args': {
            'flow': 'SNFVAE',  # 'orthogonal', 'triangular', ...
            'num_flows': 8,
            'num_ortho_vecs': 8,
            'beta': 10.0,
        },

        # setup
        'specType': 'auto',  # 'auto', 'synth', 'invivo', 'fMRSinPain', 'all'
        'ppmlim': (0.5, 4.5),  # ppm limits for the spectra (used if specType='auto')
        'basisFmt': '7tslaser',  # '', 'cha_philips', 'fMRSinPain', 'biggaba', '7tslaser', ...
        'dataType': 'aumc2',  # 'clean', 'std', 'std_rw', 'std_rw_p', 'custom', ...
        'test_size': 1000,  # number of test samples

        # for lcm model
        'method': 'Newton',  # 'Newton', 'MH', 'LCModel'
        'bandwidth': 3000,  # bandwidth of the spectra
        'sample_points': 1024,  # number of sample points
        'save_path': '',  # './testLCM/',  # path to save the results (empty for no saving)

        # mode
        'load_model': True,  # load model from path2trained
        'skip_train': True,  # skip the training procedure

        # visual settings
        'run': False,  # run the inference (will try to load results if False)
        'save': True,  # save the plots
        'path2save': './tests/results/snf/',  # path to save the results
        'error': 'msmae',  # 'mae', 'mse', 'mape', ...

        'compute_metrics': True,  # compute metrics
        'compute_metrics_prior': True,  # compute metrics for prior
        'individual_plots': False,  # plot individual results
        'pair_plot': False,  # plot pairwise results (all-in-one)
        'lcmodel_comp': False,  # compare with LCModel
        'tsne': False,  # t-SNE on latent space
        'calibration': False,  # calibration of the model
        'model_comp_mae': True,  # plot mae comparison of all models
    }

    # create the path to save the results
    config['path2save'] = os.path.join(config['path2save'], f"{config['flow_args']['flow']}_b{int(config['flow_args']['beta'])}/{config['dataType']}/")


    # initialize the test
    test = Test(config)

    param_names = (test.model.basisObj.names +
                   ['Lor. Broad.', 'Gauss. Broad.', 'Freq. Shift', 'Zero Order Phase', 'First Order Phase',
                    'Baseline Param. 1', 'Baseline Param. 2', 'Baseline Param. 3',
                    'Baseline Param. 4', 'Baseline Param. 5', 'Baseline Param. 6',])
    num_params = 20

    # run the test
    if config['run']:
        print('Running the test...', end=' ')

        x, y, t = next(iter(test.dataloader.test_dataloader()))
        x, y, t = x.to(test.model.device), y.to(test.model.device), t.to(test.model.device)

        lim = 16
        if x.shape[0] > lim:   # memory limit
            with torch.no_grad():
                xs = torch.zeros(x.shape[0], 2, test.system.last - test.system.first)
                specs = torch.zeros(config['num_samples'], x.shape[0], 2, test.system.last - test.system.first)
                zs = torch.zeros(config['num_samples'], x.shape[0], t.shape[1])
                for i in range(0, x.shape[0], lim):
                    snf = deepcopy(test.model)
                    xs_, specs_, zs_ = snf.forward_sample(x[i:i + lim].clone(), n_samples=config['num_samples'])
                    xs[i:i + lim] = xs_.detach().cpu()
                    specs[:, i:i + lim] = specs_.detach().cpu()
                    zs[:, i:i + lim] = zs_.detach().cpu()
                    del xs_, specs_, zs_, snf

        else:
            xs, specs, zs = test.model.forward_sample(x.clone(), n_samples=config['num_samples'])

        # to cpu
        x, y, t = x.detach().cpu(), y.detach().cpu(), t.detach().cpu()
        xs, specs, zs = xs.detach().cpu(), specs.detach().cpu(), zs.detach().cpu()

        # save results
        if config['save']:
            print('Saving...', end=' ')

            if not os.path.exists(config['path2save']): os.makedirs(config['path2save'])

            with h5py.File(os.path.join(config['path2save'], 'results.h5'), 'w') as f:
                f.create_dataset('x', data=x)
                f.create_dataset('xs', data=xs)
                f.create_dataset('y', data=y)
                f.create_dataset('t', data=t)
                f.create_dataset('specs', data=specs)
                f.create_dataset('zs', data=zs)
        print('Done.')

    # load results
    else:
        print('Loading results...', end=' ')
        with h5py.File(os.path.join(config['path2save'], 'results.h5'), 'r') as f:
            x = f['x'][:]
            xs = f['xs'][:]
            y = f['y'][:]
            t = f['t'][:]
            specs = f['specs'][:]
            zs = f['zs'][:]
        print('Done.')


    # process
    gts = test.system.sigModel.forward(torch.Tensor(t))
    zs[..., :test.system.basisObj.n_metabs][zs[..., :test.system.basisObj.n_metabs] < 0] = 0


    if config['compute_metrics']:
        print('Computing metrics...')

        # save path
        print(config['path2save'])

        # take mean as most probable value
        zs_mean = zs.mean(axis=0)

        # exclude macromolecules
        idx = [i for i, name in enumerate(test.system.basisObj.names)
               if 'mm' in name.lower() or 'mac' in name.lower()]
        t[:, idx] = 0
        zs_mean[:, idx] = 0

        # compute MAE
        err = test.system.concsLoss(torch.Tensor(t)[:, :test.system.basisObj.n_metabs],
                                    torch.Tensor(zs_mean)[:, :test.system.basisObj.n_metabs],
                                    type='mae')
        print('MAE (± SE):', err.mean().item(), '±', err.std().item() / np.sqrt(t.shape[0]))

        # compute lin's ccc
        lin_ccc = test.system.concsLoss(torch.Tensor(t)[:, :test.system.basisObj.n_metabs],
                                        torch.Tensor(zs_mean)[:, :test.system.basisObj.n_metabs],
                                        type='ccc')

        # remove NaN values
        lin_ccc = lin_ccc[~torch.isnan(lin_ccc)]

        print('CCC (± SE):', lin_ccc.mean().item(), '±', lin_ccc.std().item() / np.sqrt(t.shape[0]))

        # compute KL and reconstruction error
        p_xs, p_x_mean, p_z_mu, p_z_var, p_ldj, p_z0, p_zk = test.model.forward(torch.Tensor(x).to(test.model.device))

        # exclude macromolecules
        p_z_mu[:, idx] = 0
        p_z_var[:, idx] = 0
        p_z0[:, idx] = 0
        p_zk[:, idx] = 0

        tot, rec, kl = test.model.flow_loss(p_xs, p_x_mean, p_z_mu, p_z_var, p_z0, p_zk, p_ldj,
                                            type='rvdb', beta=config['flow_args']['beta'])
        print('RvdB Total Loss (± SE):', tot.mean().item(), '±', tot.std().item() / np.sqrt(t.shape[0]))
        print('RvdB Reconstruction Error (± SE):', rec.mean().item(), '±', rec.std().item() / np.sqrt(t.shape[0]))
        print('RvdB KL Divergence (± SE):', kl.mean().item(), '±', kl.std().item() / np.sqrt(t.shape[0]))

        tot_nll, rec_nll, kl_nll = test.model.flow_loss(p_xs, p_x_mean, p_z_mu, p_z_var, p_z0, p_zk, p_ldj,
                                            type='nll', beta=config['flow_args']['beta'])

        print('NLL Total Loss (± SE):', tot_nll.mean().item(), '±', tot_nll.std().item() / np.sqrt(t.shape[0]))
        print('NLL Reconstruction Error (± SE):', rec_nll.mean().item(), '±', rec_nll.std().item() / np.sqrt(t.shape[0]))
        print('NLL KL Divergence (± SE):', kl_nll.mean().item(), '±', kl_nll.std().item() / np.sqrt(t.shape[0]))

        # save metrics
        if not os.path.exists(config['path2save'] + 'metrics/'): os.makedirs(config['path2save'] + 'metrics/')
        with open(config['path2save'] + 'metrics/metrics.txt', 'w') as f:
            f.write('MAE (± SE): ' + str(err.mean().item()) + ' ± ' + str(err.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('CCC (± SE): ' + str(lin_ccc.mean().item()) + ' ± ' + str(lin_ccc.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('RvdB Total Loss (± SE): ' + str(tot.mean().item()) + ' ± ' + str(tot.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('RvdB Reconstruction Error (± SE): ' + str(rec.mean().item()) + ' ± ' + str(rec.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('RvdB KL Divergence (± SE): ' + str(kl.mean().item()) + ' ± ' + str(kl.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('NLL Total Loss (± SE): ' + str(tot_nll.mean().item()) + ' ± ' + str(tot_nll.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('NLL Reconstruction Error (± SE): ' + str(rec_nll.mean().item()) + ' ± ' + str(rec_nll.std().item() / np.sqrt(t.shape[0])) + '\n')
            f.write('NLL KL Divergence (± SE): ' + str(kl_nll.mean().item()) + ' ± ' + str(kl_nll.std().item() / np.sqrt(t.shape[0])) + '\n')

        print('Done.')


    if config['compute_metrics_prior']:
        print('Computing metrics for prior...')

        # sample from prior
        min_prior, max_prior = test.model.p_min.view(1, 1, -1), test.model.p_max.view(1, 1, -1)
        zs_prior = torch.randn(zs.shape) * (max_prior - min_prior) + min_prior
        zs_mean_prior = zs_prior.mean(axis=0)
        z_var_prior = zs_prior.var(axis=0).log()

        # exclude macromolecules
        idx = [i for i, name in enumerate(test.system.basisObj.names)
               if 'mm' in name.lower() or 'mac' in name.lower()]
        t[:, idx] = 0
        zs_mean_prior[:, idx] = 0

        # compute MAE
        err = test.system.concsLoss(torch.Tensor(t)[:, :test.system.basisObj.n_metabs],
                                    torch.Tensor(zs_mean_prior)[:, :test.system.basisObj.n_metabs],
                                    type='mae')
        print('MAE (± SE):', err.mean().item(), '±', err.std().item() / np.sqrt(t.shape[0]))

        # compute lin's ccc
        lin_ccc = test.system.concsLoss(torch.Tensor(t)[:, :test.system.basisObj.n_metabs],
                                        torch.Tensor(zs_mean_prior)[:, :test.system.basisObj.n_metabs],
                                        type='ccc')
        # remove NaN values
        lin_ccc = lin_ccc[~torch.isnan(lin_ccc)]

        print('CCC (± SE):', lin_ccc.mean().item(), '±', lin_ccc.std().item() / np.sqrt(t.shape[0]))

        # compute other metrics
        z0_prior = torch.randn((zs.shape[1], zs.shape[2])) * (max_prior - min_prior) + min_prior
        z0_prior = z0_prior.squeeze(0)

        # compute mean spectrum
        mean_spectrum = test.model.sigModel.forward(z0_prior)
        mean_spectrum = torch.stack((mean_spectrum.real, mean_spectrum.imag), dim=1)
        mean_spectrum = mean_spectrum[..., test.system.first:test.system.last] / test.model.scale

        # compute KL and reconstruction error
        tot, rec, kl = test.model.flow_loss(torch.Tensor(xs), mean_spectrum, zs_mean_prior, z_var_prior, z0_prior, z0_prior,
                                            torch.zeros(1), type='rvdb', beta=config['flow_args']['beta'])

        print('RvdB Total Loss (± SE):', tot.mean().item(), '±', tot.std().item() / np.sqrt(t.shape[0]))
        print('RvdB Reconstruction Error (± SE):', rec.mean().item(), '±', rec.std().item() / np.sqrt(t.shape[0]))
        print('RvdB KL Divergence (± SE):', kl.mean().item(), '±', kl.std().item() / np.sqrt(t.shape[0]))

        tot, rec, kl = test.model.flow_loss(torch.Tensor(xs), mean_spectrum, zs_mean_prior, z_var_prior, z0_prior, z0_prior,
                                            torch.zeros(1), type='nll', beta=config['flow_args']['beta'])

        print('NLL Total Loss (± SE):', tot.mean().item(), '±', tot.std().item() / np.sqrt(t.shape[0]))
        print('NLL Reconstruction Error (± SE):', rec.mean().item(), '±', rec.std().item() / np.sqrt(t.shape[0]))
        print('NLL KL Divergence (± SE):', kl.mean().item(), '±', kl.std().item() / np.sqrt(t.shape[0]))

        print('Done.')


    if config['individual_plots']:
        print('Plotting individuals...', end=' ')

        if config['ex_idx'] == []: config['ex_idx'] = list(range(x.shape[0]))
        elif isinstance(config['ex_idx'], int): config['ex_idx'] = [config['ex_idx']]

        for idx in config['ex_idx']:

            # example data
            ex_spectrum = xs[idx, 0, :].numpy() if isinstance(x, torch.Tensor) else xs[idx, 0, :]
            fits = specs[:, idx, 0, :].numpy() if isinstance(specs, torch.Tensor) else specs[:, idx, 0, :]
            zk = zs[:, idx, :num_params].numpy() if isinstance(zs, torch.Tensor) else zs[:, idx, :num_params]
            ti = t[idx, :num_params].numpy() if isinstance(t, torch.Tensor) else t[idx, :num_params]
            gt = gts[idx].real.numpy() if isinstance(gts, torch.Tensor) else gts[idx].real
            gt = gt[test.system.first:test.system.last] / test.model.scale

            # ========== plot spectra and fits ==========
            plt.figure(figsize=(7, 3))

            for i, fit in enumerate(fits):
                plt.plot(test.model.basisObj.ppm[test.model.first:test.model.last], fit,
                         color='purple', alpha=0.2, linewidth=1, label='SNF Fit' if i == 0 else None)

                res = ex_spectrum - fit
                plt.plot(test.model.basisObj.ppm[test.model.first:test.model.last],
                         res + 1.2 * ex_spectrum.max(),
                         color='grey', alpha=0.2, linewidth=0.5, label='Residuals' if i == 0 else None)

            plt.plot(test.model.basisObj.ppm[test.model.first:test.model.last], ex_spectrum,
                     label='Measured Spectrum', color='black', linewidth=2)

            plt.plot(test.model.basisObj.ppm[test.model.first:test.model.last], gt,
                     label='Ground Truth', color='blue', linewidth=1, linestyle='--')


            # plt.title('Example Spectrum and Fits')
            plt.xlabel('Chemical Shift [ppm]')
            # plt.ylabel('Amplitude')
            # lgd = plt.legend(loc='upper right', bbox_to_anchor=(1, 1.3))
            # plt.grid(alpha=0.5)
            plt.gca().invert_xaxis()
            plt.tight_layout()

            # no y-axis ticks and labels
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])

            if config['save']:
                if not os.path.exists(config['path2save'] + 'fits/'): os.makedirs(config['path2save'] + 'fits/')
                plt.savefig(config['path2save'] + 'fits/' + f'example_{idx}.svg', dpi=300,
                            bbox_inches='tight')#, bbox_extra_artists=(lgd,))
                plt.close()
            else:
                plt.show()


            # ========== latent space posterior ==========
            # compute min and max distance from mean for all parameters
            min_dist = np.min(zk - zk.mean(axis=0, keepdims=True)) - 0.1
            max_dist = np.max(zk - zk.mean(axis=0, keepdims=True)) + 0.1

            # print(min_dist, max_dist)

            fig, axes = plt.subplots(num_params // 5, 5, figsize=(15, 12))  # adjust grid size if needed
            axes = axes.flatten()

            for i in range(num_params):
                data = zk[:, i]

                # determine the number of unique values to check if all values are (nearly) identical
                unique_vals = np.unique(data)

                if len(unique_vals) == 1:
                    # if only one unique value, center a small bin around it
                    center = unique_vals[0]
                    binwidth = 0.01 * (max_dist - min_dist)  # adjust bin width based on range
                    bins = np.arange(center - binwidth, center + binwidth, binwidth / 2)
                else:
                    # default bin setting for normal cases
                    bins = 'auto'

                sns.histplot(data, kde=True, ax=axes[i], color='purple', alpha=0.5, bins=bins)

                axes[i].set_title(param_names[i], fontsize=12)
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
                axes[i].grid(alpha=0.3)

                # add ground truth
                axes[i].axvline(ti[i], color='blue', linestyle='--', label='Ground Truth')

                # limit x-axis based on mean-centered range
                axes[i].set_xlim(data.mean() + min_dist, data.mean() + max_dist)

            # hide empty subplots
            for j in range(num_params, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()

            if config['save']:
                if not os.path.exists(config['path2save'] + 'latent/'): os.makedirs(config['path2save'] + 'latent/')
                plt.savefig(config['path2save'] + 'latent/' + f'latent_{idx}.svg', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()


            # ========== per-sample posterior mean vs ground Truth ==========
            posterior_means = zk.mean(axis=0)
            posterior_stds = zk.std(axis=0)

            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(num_params), posterior_means, yerr=posterior_stds, capsize=5, alpha=0.6,
                    label='Posterior Mean ± Std')
            plt.scatter(np.arange(num_params), ti, color='red', label='Ground Truth', zorder=10)
            plt.xticks(np.arange(num_params), param_names[:num_params], rotation=45, ha='right')
            plt.ylabel('Parameter Value')
            plt.title(f'Posterior Estimates vs. Ground Truth (Sample {idx})')
            plt.legend()
            plt.tight_layout()

            if config['save']:
                if not os.path.exists(config['path2save'] + 'posterior_bar/'): os.makedirs(
                    config['path2save'] + 'posterior_bar/')
                plt.savefig(config['path2save'] + 'posterior_bar/' + f'bar_{idx}.svg', dpi=300)
                plt.close()
            else:
                plt.show()


            # ========== residual envelope (overlayed fits) ==========
            residuals = ex_spectrum - fits  # [num_samples, spectrum_len]
            res_mean = residuals.mean(axis=0)
            res_std = residuals.std(axis=0)

            plt.figure(figsize=(10, 4))
            ppm = test.model.basisObj.ppm[test.model.first:test.model.last]

            plt.plot(ppm, res_mean, label='Mean Residual', color='darkred')
            plt.fill_between(ppm, res_mean - res_std, res_mean + res_std, alpha=0.3, color='red', label='±1 STD')
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.title(f'Residual Envelope (Sample {idx})')
            plt.xlabel('Chemical Shift [ppm]')
            plt.ylabel('Residual Amplitude')
            plt.gca().invert_xaxis()
            plt.legend()
            plt.tight_layout()

            if config['save']:
                if not os.path.exists(config['path2save'] + 'residual_envelope/'): os.makedirs(
                    config['path2save'] + 'residual_envelope/')
                plt.savefig(config['path2save'] + 'residual_envelope/' + f'resid_{idx}.svg', dpi=300)
                plt.close()
            else:
                plt.show()


            # ========== posterior correlation per parameter (scatter plots) ==========
            corr_matrix = np.corrcoef(zk.T)

            # plot scatter matrix (showing correlation between parameters)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=param_names[:num_params],
                        yticklabels=param_names[:num_params])
            plt.title(f'Posterior Correlation Matrix (Sample {idx})')
            plt.tight_layout()

            if config['save']:
                if not os.path.exists(config['path2save'] + 'posterior_corr/'): os.makedirs(
                    config['path2save'] + 'posterior_corr/')
                plt.savefig(config['path2save'] + 'posterior_corr/' + f'corr_{idx}.svg', dpi=300)
                plt.close()
            else:
                plt.show()


            # ========== PCA on latent space (per sample) ==========
            pca = PCA(n_components=2)
            latent_pca = pca.fit_transform(zk)

            # plot PCA results
            plt.figure(figsize=(8, 6))
            plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6, color='purple')
            plt.title(f'PCA of Latent Space (Sample {idx})', fontsize=16)
            plt.xlabel('PCA Component 1', fontsize=14)
            plt.ylabel('PCA Component 2', fontsize=14)
            plt.tight_layout()

            if config['save']:
                if not os.path.exists(config['path2save'] + 'pca/'): os.makedirs(config['path2save'] + 'pca/')
                plt.savefig(config['path2save'] + 'pca/' + f'pca_{idx}.svg', dpi=300)
                plt.close()
            else:
                plt.show()


            # ========== t-SNE on latent space (per sample) ==========
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            latent_tsne = tsne.fit_transform(zk)

            # plot t-SNE results
            plt.figure(figsize=(8, 6))
            plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6, color='purple')
            plt.title(f't-SNE of Latent Space (Sample {idx})', fontsize=16)
            plt.xlabel('t-SNE Component 1', fontsize=14)
            plt.ylabel('t-SNE Component 2', fontsize=14)
            plt.tight_layout()

            if config['save']:
                if not os.path.exists(config['path2save'] + 'tsne/'): os.makedirs(config['path2save'] + 'tsne/')
                plt.savefig(config['path2save'] + 'tsne/' + f'tsne_{idx}.svg', dpi=300)
                plt.close()
            else:
                plt.show()
        print('Done.')


    if config['pair_plot']:
        print('Plotting all-in-one pair-plot...', end=' ')

        def plot_fancy_pairplot(zs, param_names, example_idx, path2save):
            """
            Create a pairplot (scatter matrix with regression lines and marginal KDEs)
            of the latent space for a given example index.

            Parameters:
              - zs: Latent samples, shape [n_samples, batch, num_params]
              - param_names: List of parameter names (for labeling)
              - example_idx: Index of the example (within the batch) to visualize.
              - path2save: Directory to save the plot.
            """
            # Convert the latent variables for the given example into a DataFrame.
            latent_data = zs[:, example_idx, :]
            df = pd.DataFrame(latent_data, columns=param_names[:latent_data.shape[1]])

            # Create a pairplot using seaborn:
            sns.set_context("talk")
            pair_grid = sns.pairplot(df, kind='reg', diag_kind='kde',
                                     plot_kws={'line_kws': {'color': 'red'},
                                               'scatter_kws': {'color': 'purple', 'alpha': 0.2, 's': 2}},
                                     diag_kws={'color': 'purple', 'alpha': 0.2, 'fill': True})
            pair_grid.fig.suptitle('Pairwise Relationships of Latent Variables', fontsize=18, y=1.02)
            plt.tight_layout()

            save_fname = os.path.join(path2save, f'pairplot_example_{example_idx}.svg')
            plt.savefig(save_fname, dpi=300, bbox_inches='tight')
            plt.close()

        # produce pairplot for the first example
        for idx in config['ex_idx']:
            if config['save']:
                if not os.path.exists(config['path2save'] + 'pairplot/'): os.makedirs(config['path2save'] + 'pairwise_all/')
            plot_fancy_pairplot(zs, param_names, example_idx=idx, path2save=config['path2save'] + 'pairwise_all/')

        print('Done.')


    if config['lcmodel_comp']:

        # LCModel comparison
        lcmodel_config = test.config.copy()
        lcmodel_config['model'] = 'lcm'
        lcmodel_config['method'] = 'LCModel'
        lcmodel_config['save_path'] = config['path2save'] + 'lcmodel/'

        lcmodel = test.getModel(lcmodel_config)
        t_lcm, u_lcm = lcmodel(x[config['ex_idx']])

        # sort the LCModel results
        t_lcm = torch.stack([torch.Tensor(t_lcm)[:, lcmodel.basisFSL.names.index(m)]
                             for m in test.system.basisObj.names], dim=1)
        u_lcm = torch.stack([torch.Tensor(u_lcm)[:, lcmodel.basisFSL.names.index(m)]
                             for m in test.system.basisObj.names], dim=1)

        # back to sd from perc
        u_lcm = u_lcm * t_lcm / 100

        # # ignore macromolecular parameters
        # idx = [i for i, name in enumerate(test.system.basisObj.names)
        #        if 'mm' in name.lower() or 'mac' in name.lower()]
        # t[:, idx] = 0
        # t_lcm[:, idx] = 0
        #
        # # scale the LCModel results
        # w = test.system.optimalReference(torch.Tensor(t)[:t_lcm.shape[0], :t_lcm.shape[1]], t_lcm, type='scipy_l1')
        # t_lcm = t_lcm * w
        # u_lcm = u_lcm * w

        print('LCModel comparison...', end=' ')

        def pairwise_latent_plot_with_refs(zs, t, t_lcm, u_lcm, param_names, limits=None,
                                           example_idx=0, param_idx=None, save_path=None):
            """
            Plots pairwise scatter plots of the latent space posterior samples,
            with ground truth (x), LCModel estimate (o), CRLBs (error bars),
            and prior limits (dashed lines/boxes).
            """
            Z = zs[:, example_idx, :]
            t_i = t[example_idx, :]
            t_lcm_i = t_lcm[example_idx, :]
            u_lcm_i = u_lcm[example_idx, :]

            n_params = len(param_idx)
            fig, axes = plt.subplots(n_params, n_params, figsize=(2.0 * n_params, 2.0 * n_params))

            for i, j in itertools.product(range(n_params), repeat=2):
                ax = axes[i, j]
                x_idx = param_idx[j]
                y_idx = param_idx[i]

                # extract the relevant data
                x_post = Z[:, x_idx]
                y_post = Z[:, y_idx]
                x_gt = t_i[x_idx].item()
                y_gt = t_i[y_idx].item()
                x_lcm = t_lcm_i[x_idx].item()
                y_lcm = t_lcm_i[y_idx].item()
                x_crlb = u_lcm_i[x_idx].item()
                y_crlb = u_lcm_i[y_idx].item()

                # prior limits
                if limits is not None:
                    x_prior_min, x_prior_max = limits[0][x_idx], limits[1][x_idx]
                    y_prior_min, y_prior_max = limits[0][y_idx], limits[1][y_idx]
                else:
                    x_prior_min, x_prior_max = x_post.min(), x_post.max()
                    y_prior_min, y_prior_max = y_post.min(), y_post.max()

                # compute full axis range including all relevant info
                x_min = min(x_post.min(), x_gt, x_lcm, x_prior_min, x_lcm - x_crlb) - 0.1 * (x_post.max() - x_post.min())
                x_max = max(x_post.max(), x_gt, x_lcm, x_prior_max, x_lcm + x_crlb) + 0.1 * (x_post.max() - x_post.min())
                y_min = min(y_post.min(), y_gt, y_lcm, y_prior_min, y_lcm - y_crlb) - 0.1 * (y_post.max() - y_post.min())
                y_max = max(y_post.max(), y_gt, y_lcm, y_prior_max, y_lcm + y_crlb) + 0.1 * (y_post.max() - y_post.min())

                if i == j:
                    # diagonal (marginal)
                    sns.histplot(x_post, kde=True, ax=ax, color='purple', alpha=0.2, bins='auto')
                    ax.axvline(x_gt, color='blue', linestyle='--', label='GT' if j == 0 else "")
                    ax.axvline(x_lcm, color='chocolate', linestyle='-', label='LCModel' if j == 0 else "")
                    ax.fill_betweenx([-np.inf, np.inf], x_lcm - x_crlb, x_lcm + x_crlb, color='chocolate', alpha=0.2)
                    # prior bounds as vertical dashed lines
                    ax.axvline(x_prior_min, color='black', linestyle='dashed', label='Prior' if j == 0 else "",
                               linewidth=1)
                    ax.axvline(x_prior_max, color='black', linestyle='dashed', linewidth=1)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylabel('')
                else:
                    # off-diagonal (scatter)
                    ax.scatter(x_post, y_post, alpha=0.2, color='purple', s=10,
                               label='Posterior' if (i == 1 and j == 0) else "")
                    ax.scatter(x_gt, y_gt, color='blue', marker='x', s=100,
                               label='GT' if (i == 1 and j == 0) else "")
                    ax.errorbar(x_lcm, y_lcm, xerr=x_crlb, yerr=y_crlb, elinewidth=1.5, capsize=3.0,
                                fmt='o', color='chocolate', label='LCModel' if (i == 1 and j == 0) else "")
                    # prior bounds as a dashed rectangle
                    rect = Rectangle((x_prior_min, y_prior_min), x_prior_max - x_prior_min, y_prior_max - y_prior_min,
                                     linewidth=1, edgecolor='black', linestyle='dashed', facecolor='none',
                                     label='Prior' if (i == 1 and j == 0) else "")
                    ax.add_patch(rect)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)

                # axis labels
                if i == n_params - 1:
                    ax.set_xlabel(param_names[x_idx])
                else:
                    ax.set_xticks([])
                if j == 0:
                    ax.set_ylabel(param_names[y_idx])
                else:
                    ax.set_yticks([])

            # handle legend
            handles, labels = axes[1, 0].get_legend_handles_labels()
            ldg = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.1))

            plt.tight_layout(pad=0.6)
            if save_path:
                if not os.path.exists(save_path): os.makedirs(save_path)
                plt.savefig(os.path.join(save_path, f'pairwise_with_refs_{example_idx}.svg'), dpi=300,
                            bbox_extra_artists=(ldg,), bbox_inches='tight')
                plt.close()
            else:
                plt.show()


        # save all specs and fits for quality check in one folder
        def plot_LCModel_fit(fit, gt, xs, idx, save_path=None):
            # plt.figure()
            # plt.plot(fit['ppm'], fit['data'], 'k', label='Data', linewidth=1)
            # plt.plot(fit['ppm'], fit['completeFit'], 'r', label='Fit', alpha=0.6, linewidth=2)
            # plt.plot(fit['ppm'], fit['data'] - fit['completeFit'] + 1.1 * np.max(fit['data']),
            #          'k', label='Residual', alpha=0.8, linewidth=1)
            # plt.xlabel('Chemical Shift [ppm]')
            # plt.gca().invert_xaxis()

            ppm = fit['ppm']
            ex_spectrum = fit['data']
            lcmodel = fit['completeFit']

            # gt scale from lcmodel to reality
            scale = xs.mean() / ex_spectrum.mean()
            ex_spectrum *= scale
            lcmodel *= scale

            plt.figure(figsize=(7, 3))

            # plt.plot(ppm, ex_spectrum,
            #          label='Measured Spectrum', color='black', linewidth=2)
            plt.plot(test.model.basisObj.ppm[test.model.first:test.model.last], xs,
                     label='Measured Spectrum', color='black', linewidth=2)

            plt.plot(test.model.basisObj.ppm[test.model.first:test.model.last], gt,
                     label='Ground Truth', color='blue', linewidth=1, linestyle='--')

            plt.plot(ppm, lcmodel,
                     label='LCModel Fit', color='chocolate', linewidth=1)

            res = ex_spectrum - lcmodel
            plt.plot(ppm, res + 1.2 * ex_spectrum.max(),
                     color='grey', alpha=0.9, linewidth=0.5, label='Residuals')

            # plt.title('Example Spectrum and Fits')
            plt.xlabel('Chemical Shift [ppm]')
            # plt.ylabel('Amplitude')
            # lgd = plt.legend(loc='upper right', bbox_to_anchor=(1, 1.3))
            # plt.grid(alpha=0.5)
            plt.gca().invert_xaxis()
            plt.tight_layout()

            # no y-axis ticks and labels
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])

            if save_path:
                if not os.path.exists(save_path): os.makedirs(save_path)
                plt.savefig(os.path.join(save_path, f'LCModel_fit_{idx}.svg'), dpi=300,
                            bbox_inches='tight')#, bbox_extra_artists=(lgd,))
                plt.close()
            else:
                plt.show()


        if config['ex_idx'] == []: config['ex_idx'] = list(range(x.shape[0]))
        elif isinstance(config['ex_idx'], int): config['ex_idx'] = [config['ex_idx']]

        for idx in config['ex_idx']:

            # example data
            ex_spectrum = xs[idx, 0, :].numpy() if isinstance(x, torch.Tensor) else xs[idx, 0, :]
            fits = specs[:, idx, 0, :].numpy() if isinstance(specs, torch.Tensor) else specs[:, idx, 0, :]
            zk = zs[:, idx, :num_params].numpy() if isinstance(zs, torch.Tensor) else zs[:, idx, :num_params]
            ti = t[idx, :num_params].numpy() if isinstance(t, torch.Tensor) else t[idx, :num_params]
            gt = gts[idx].real.numpy() if isinstance(gts, torch.Tensor) else gts[idx].real
            gt = gt[test.system.first:test.system.last] / test.model.scale

            # ========== plot pairwise each parameter ==========
            # adding ground truth, LCModel estimate, and CRLB
            # metabs = ['Cr', 'GABA', 'Glu', 'GPC', 'mIns', 'NAA', 'PCh', 'PCr']
            metabs = ['Cr', 'GABA', 'Glu', 'GPC', 'NAA', 'PCh', 'PCr']
            # metabs = ['Cr', 'GABA', 'Glu', 'mIns', 'NAA', 'PCr']
            metab_idx = [test.system.basisObj.names.index(m) for m in metabs]

            pairwise_latent_plot_with_refs(zs, t, t_lcm, u_lcm, param_names,
                                           limits=(test.model.p_min, test.model.p_max),
                                           example_idx=idx, param_idx=metab_idx,
                                           save_path=f"{config['path2save']}pairwise/")


            # ========== plot LCModel fit ==========
            fit = lcmodel.read_LCModel_fit(f"{lcmodel_config['save_path']}temp{idx}.coord")
            plot_LCModel_fit(fit, gt, ex_spectrum, idx, save_path=f"{config['path2save']}lcmodel_fits/")

        print('Done.')


    if config['tsne']:
        print('t-SNE on latent space...', end=' ')

        zs_sne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(zs.reshape(-1, zs.shape[-1]))
        zs_sne = zs_sne.reshape(zs.shape[0], zs.shape[1], 2)

        # ========== plot t-SNE results with different colors for each example ==========
        plt.figure(figsize=(8, 6))
        for idx in config['ex_idx']:
            plt.scatter(zs_sne[:, idx, 0], zs_sne[:, idx, 1], alpha=0.6, label=f'Example {idx + 1}')
        plt.title('t-SNE of Latent Space', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if config['save']:
            if not os.path.exists(config['path2save']): os.makedirs(config['path2save'])
            plt.savefig(config['path2save'] + f'tsne.svg', dpi=300)
            plt.close()
        else:
            plt.show()


        #  ========== plot t-SNE results with different colors for each parameter ==========
        plt.figure(figsize=(8, 6))
        for i in range(num_params):
            plt.scatter(zs_sne[i, :, 0], zs_sne[i, :, 1], alpha=0.3, label=param_names[i], s=5)
        plt.title('t-SNE of Latent Space', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if config['save']:
            if not os.path.exists(config['path2save']): os.makedirs(config['path2save'])
            plt.savefig(config['path2save'] + f'tsne_params.svg', dpi=300)
            plt.close()
        else:
            plt.show()

        print('Done.')


    if config['calibration']:
        print('Calibration evaluation...', end=' ')

        def evaluate_calibration_and_uncertainty(zs, t, param_names, num_params, quantile_levels=None, model=''):
            """
            Evaluate calibration and sharpness for probabilistic regression.

            Parameters:
                zs: np.ndarray of shape (S, B, M) - sampled predictions (S samples)
                t: np.ndarray of shape (B, M)     - ground truth targets
                param_names: list of parameter names
                num_params: int - number of parameters
                quantile_levels: list of percentiles for CIs (e.g., [50, 90, 95])

            Returns:
                coverage_table: pd.DataFrame
                calibration_plot: matplotlib figure
            """
            S, B, M = zs.shape
            if quantile_levels is None:
                quantile_levels = [50, 90, 95]

            # convert to numpy if needed
            zs = np.asarray(zs)
            t = np.asarray(t)

            # store results
            coverage_results = {}
            width_results = {}

            nominal_coverages = np.linspace(0.01, 0.99, 20)
            empirical_coverages = []

            for nominal in nominal_coverages:
                lower_q = (1 - nominal) / 2 * 100
                upper_q = (1 + nominal) / 2 * 100
                lower = np.percentile(zs, lower_q, axis=0)  # (B, M)
                upper = np.percentile(zs, upper_q, axis=0)
                covered = (t >= lower) & (t <= upper)  # (B, M)
                empirical_coverages.append(covered.mean(axis=0))  # list of (M,)

            empirical_coverages = np.stack(empirical_coverages, axis=0)  # (N_levels, M)

            # plot calibration curve
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            for m in range(min(num_params, M)):
                ax.plot(nominal_coverages, empirical_coverages[:, m], label=f"{param_names[m]}", alpha=0.7)
            ax.plot([0, 1], [0, 1], 'k--', label="Ideal")
            ax.set_xlabel("Nominal Coverage")
            ax.set_ylabel("Empirical Coverage")
            ax.set_title(f"{model.replace('VAE', '') if not 'base' in model else model.replace('base', '')} Calibration Curve")
            # lgd = ax.legend(loc='lower right', fontsize='small', bbox_to_anchor=(1.3, -0.15))
            lgd = None
            ax.grid(True)

            # create summary table
            summary = {"Metabolite": []}
            for q in quantile_levels:
                summary[f"{q}% CI Coverage"] = []
                summary[f"{q}% CI Width"] = []

                lower_q = (100 - q) / 2
                upper_q = 100 - lower_q
                lower = np.percentile(zs, lower_q, axis=0)  # (B, M)
                upper = np.percentile(zs, upper_q, axis=0)

                covered = (t >= lower) & (t <= upper)  # (B, M)
                width = upper - lower  # (B, M)

                for m in range(M):
                    summary[f"{q}% CI Coverage"].append(covered[:, m].mean())
                    summary[f"{q}% CI Width"].append(width[:, m].mean())

            summary["Metabolite"] = [f"Metab {m}" for m in range(M)]
            df_summary = pd.DataFrame(summary)

            return df_summary, fig, lgd

        # evaluate calibration and uncertainty
        quantile_levels = [50, 90, 95]
        df_summary, fig, lgd = evaluate_calibration_and_uncertainty(zs, t, param_names, num_params, quantile_levels,
                                                                    model=config['flow_args']['flow'])

        # save calibration plot
        if config['save']:
            if not os.path.exists(config['path2save']): os.makedirs(config['path2save'])
            fig.savefig(os.path.join(config['path2save'], 'calibration_plot.svg'), dpi=300,
                        bbox_inches='tight')#, bbox_extra_artists=(lgd,))
            plt.close()
        else:
            plt.show()


        if config['lcmodel_comp']:
            zs_lcmodel = np.random.normal(loc=t_lcm[None, :, :], scale=np.clip(u_lcm[None, :, :], 1e-4, None),
                                          size=(zs.shape[0],) + t_lcm.shape)

            df_summary, fig, lgd = evaluate_calibration_and_uncertainty(zs_lcmodel, t[:t_lcm.shape[0], :t_lcm.shape[1]],
                                                                        param_names, num_params, quantile_levels, model="LCModel")

            # save calibration plot
            if config['save']:
                if not os.path.exists(config['path2save']): os.makedirs(config['path2save'])
                fig.savefig(os.path.join(config['path2save'], 'calibration_plot_lcmodel.svg'), dpi=300,
                            bbox_inches='tight')#, bbox_extra_artists=(lgd,))
                plt.close()
            else:
                plt.show()

        print('Done.')

        # print(df_summary)


    if config['model_comp_mae']:
        print('Model comparison (MAE, Variance, ...)', end=' ')

        # model paths
        model_paths = {
            'SNF': f"./tests/results/snf/SNFVAE_b{int(config['flow_args']['beta'])}/{config['dataType']}/results.h5",
            'VAE': f"./tests/results/snf/baseVAE_b{int(config['flow_args']['beta'])}/{config['dataType']}/results.h5",
            # 'IAF': f"./tests/results/snf/IAFVAE_b{int(config['flow_args']['beta'])}/{config['dataType']}/results.h5",
            'LCModel': f"./tests/results/snf/SNFVAE_b{int(config['flow_args']['beta'])}/{config['dataType']}/lcmodel/",
        }

        models = list(model_paths.keys())
        # colors = ['purple', 'tab:blue', 'tab:red', 'chocolate']
        colors = ['purple', 'tab:blue', 'chocolate']
        snr_all, lw_all, mae_all, std_all = {}, {}, {}, {}
        mm_all, pha_all = {}, {}

        # save t and y
        t_save = t.copy()
        y_save = y.copy()

        for model in models:

            t = t_save.copy()
            y = y_save.copy()

            try:
                if model == 'LCModel':
                    lcmodel_config = test.config.copy()
                    lcmodel_config['model'] = 'lcm'
                    lcmodel_config['method'] = 'LCModel'
                    lcmodel = test.getModel(lcmodel_config)

                    # from filepath load lcmodel results
                    zs_mean, zs_std = [], []
                    for filename in os.listdir(model_paths[model]):
                        if filename.endswith('.coord'):
                            # read .coord file
                            metabs, concs, crlbs, tcr = lcmodel.read_LCModel_coord(model_paths[model] + filename,
                                                                                   meta=False)
                            # sort concentrations by basis names
                            t_lcm = [concs[metabs.index(met)] if met in metabs else 0.0
                                     for met in lcmodel.basisFSL._names]
                            u_lcm = [crlbs[metabs.index(met)] if met in metabs else 999.0
                                     for met in lcmodel.basisFSL._names]

                            zs_mean.append(t_lcm)
                            zs_std.append(u_lcm)

                    # sort the LCModel results
                    zs_mean = torch.stack([torch.Tensor(zs_mean)[:, lcmodel.basisFSL.names.index(m)]
                                           for m in test.system.basisObj.names], dim=1)
                    zs_std = torch.stack([torch.Tensor(zs_std)[:, lcmodel.basisFSL.names.index(m)]
                                          for m in test.system.basisObj.names], dim=1)

                    # back to sd from perc
                    zs_std = zs_std * zs_mean / 100

                else:
                    try:  # loading other models
                        with h5py.File(model_paths[model], 'r') as f:
                            zs = np.array(f['zs'])  # latent samples (100, batch, metab_dim)
                            t = np.array(f['t'])  # true concentrations
                            y = np.array(f['y'])  # separated specs: [..., :N] = signal, [..., -1] = noise
                        zs_mean, zs_std = zs.mean(0), zs.std(0)

                    except:  # run other models
                        model_config = test.config.copy()
                        model_config['model'] = model
                        model_config['method'] = 'VAE'
                        model_config['save_path'] = config['path2save'] + f'{model}/'

                        model_instance = test.getModel(model_config)
                        zs, t, y = model_instance(x)

                        # save results
                        if not os.path.exists(model_config['save_path']): os.makedirs(model_config['save_path'])
                        with h5py.File(model_config['save_path'] + 'results.h5', 'w') as f:
                            f.create_dataset('zs', data=zs.numpy())
                            f.create_dataset('t', data=t.numpy())
                            f.create_dataset('y', data=y.numpy())
                        zs_mean, zs_std = zs.mean(0), zs.std(0)

            except Exception as e:
                print(f"\n[Warning] Skipping {model}: {e}")
                continue

            # ignore macromolecular parameters
            idx = [i for i, name in enumerate(test.system.basisObj.names)
                   if 'mm' in name.lower() or 'mac' in name.lower()]
            t[:, idx] = 0
            zs_mean[:, idx] = 0

            # compute MAE
            mae = test.system.concsLoss(torch.Tensor(t)[:, :test.system.basisObj.n_metabs].clone(),
                                        torch.Tensor(zs_mean)[:, :test.system.basisObj.n_metabs],
                                        type='mae')

            print(f"MAE for {model}: {mae.mean():.4f} ± {mae.std() / np.sqrt(len(mae)):.4f}")

            # ccc
            ccc = test.system.concsLoss(torch.Tensor(t)[:, :test.system.basisObj.n_metabs],
                                        torch.Tensor(zs_mean)[:, :test.system.basisObj.n_metabs],
                                        type='ccc')
            print(f"CCC for {model}: {ccc.mean():.4f} ± {ccc.std() / np.sqrt(len(ccc)):.4f}")

            mae_all[model] = mae
            std_all[model] = zs_std[:, :test.system.basisObj.n_metabs]

            # compute SNR
            signal = y[..., :-1].sum(axis=-1)
            signal = signal[:, 0] + 1j * signal[:, 1]
            signal = signal[:, test.system.first:test.system.last]
            signal_power = np.sum(np.abs(signal) ** 2, axis=-1)

            noise = y[..., -1]
            noise = noise[:, 0] + 1j * noise[:, 1]
            noise = noise[:, test.system.first:test.system.last]
            noise_power = np.sum(np.abs(noise) ** 2, axis=-1)

            snr = 10 * np.log10(signal_power / noise_power)
            snr_all[model] = snr
            print(f"Min SNR for {model}: {snr.min():.4f} dB, Max SNR: {snr.max():.4f} dB")

            # compute linewidth
            linewidth = t[:, test.system.basisObj.n_metabs] + t[:, test.system.basisObj.n_metabs + 1]
            lw_all[model] = linewidth

            # add macromolecular parameters
            mm_all[model] = t[:, idx].sum(axis=1)

            # add phase parameters
            pha_all[model] = t[:, test.system.basisObj.n_metabs + 3]


        # def smooth_by_snr_bins(param, mae, std, bin_width=3):
        #     df = pd.DataFrame({'param': param, 'mae': mae, 'std': std})
        #     df['param_bin'] = (df['param'] // bin_width) * bin_width
        #     grouped = df.groupby('param_bin').agg({
        #         'param': 'mean', 'mae': 'mean', 'std': 'mean'
        #     }).dropna()
        #     return grouped['param'].values, grouped['mae'].values, grouped['std'].values

        def smooth_by_snr_bins(param, mae, std, bin_width=3):
            df = pd.DataFrame({'param': param, 'mae': mae, 'std': std})
            df['param_bin'] = (df['param'] // bin_width) * bin_width
            grouped = df.groupby('param_bin').agg({
                'param': 'mean',
                'mae': 'mean',
                'std': 'mean',
                'param_bin': 'count'  # this gives bin counts
            }).rename(columns={'param_bin': 'count'}).dropna()

            return (
                grouped['param'].values,
                grouped['mae'].values,
                grouped['std'].values,
                grouped['count'].values
            )

        def plot_comparison(param, param_names, num_params, models, colors, xlabel, ylabel, save_path):
            for idx, metab in enumerate(param_names):
                if idx > num_params: break
                plt.figure(figsize=(4, 3.5))
                for model, color in zip(reversed(models), reversed(colors)):
                    if model in param:
                        param_sm, mae_sm, std_sm, c = smooth_by_snr_bins(param[model],
                                                                      mae_all[model][:, idx],
                                                                      std_all[model][:, idx])
                        # param_sm, mae_sm, std_sm = param[model], mae_all[model][:, idx], std_all[model][:, idx]

                        mask = c >= 3
                        param_sm, mae_sm, std_sm = param_sm[mask], mae_sm[mask], std_sm[mask]

                        plt.plot(param_sm, mae_sm, label=model, color=color)
                        plt.fill_between(param_sm,
                                         mae_sm - std_sm,
                                         mae_sm + std_sm,
                                         color=color, alpha=0.3)

                # set x-axis limits
                plt.xlim(np.max([p.min() for p in param.values()]),
                         np.min([p.max() for p in param.values()]))

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(metab)

                # # legend outside the plot
                # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

                plt.grid(True)

                if config['save']:
                    if not os.path.exists(save_path): os.makedirs(save_path)
                    plt.savefig(save_path + f'model_comp_{metab}.svg', dpi=300,
                                bbox_inches='tight')
                else:
                    plt.show()


        # ========== SNR vs MAE plot ==========
        plot_comparison(snr_all, param_names, num_params, models, colors,
                        xlabel='SNR [dB]', ylabel='MAE ± STD',
                        save_path=config['path2save'] + 'model_comp_snr/')


        # ==========  Linewidth vs MAE plot ==========
        plot_comparison(lw_all, param_names, num_params, models, colors,
                        xlabel='Linewidth [Hz]', ylabel='MAE ± STD',
                        save_path=config['path2save'] + 'model_comp_lw/')


        # ==========  MM vs MAE plot ==========
        plot_comparison(mm_all, param_names, num_params, models, colors,
                        xlabel='Macromolecular Concentration [mM]', ylabel='MAE ± STD',
                        save_path=config['path2save'] + 'model_comp_mm/')

        # ==========  Phase vs MAE plot ==========
        plot_comparison(pha_all, param_names, num_params, models, colors,
                        xlabel='Phase [rad]', ylabel='MAE ± STD',
                        save_path=config['path2save'] + 'model_comp_phase/')

        print('Done.')