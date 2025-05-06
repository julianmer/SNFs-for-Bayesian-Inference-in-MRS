####################################################################################################
#                                         frameworkFSL.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 08/10/23                                                                                #
#                                                                                                  #
# Purpose: FSL-MRS optimization framework for least-squares fitting of spectra.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import multiprocessing
import numpy as np
import os
import pickle

from fsl_mrs.core import MRS
from fsl_mrs.utils import fitting, mrs_io, quantify

from scipy.optimize import minimize



#**************************************************************************************************#
#                                        Class FrameworkFSL                                        #
#**************************************************************************************************#
#                                                                                                  #
# The framework wrapper for the FSL-MRS fitting tool.                                              #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkFSL():
    def __init__(self, path2basis, method='Newton', multiprocessing=False, ppmlim=(0.5, 4.2),
                 conj=False, unc='perc', save_path='', bandwidth=None, sample_points=None, 
                 TE=None, TR=None, nucleus='1H', include_params=False, **kwargs):

        self.method = method   # 'Newton', 'MH'
        self.basisFSL = mrs_io.read_basis(path2basis)
        if bandwidth is not None and sample_points is not None:
            self.basisFSL._raw_fids = self.basisFSL.get_formatted_basis(bandwidth, sample_points)
            self.basisFSL._dt = 1. / bandwidth
        self.multiprocessing = multiprocessing
        self.ppmlim = ppmlim
        self.conj = conj
        self.unc = unc   # 'perc' (= clrb%), 'crlb'
        self.save_path = save_path
        self.TE, self.TR = TE, TR
        self.nucleus = nucleus
        self.include_params = include_params   # if true also output the signal parameters


    #**********************#
    #   forward function   #
    #**********************#
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    #****************************#
    #   loss on concentrations   #
    #****************************#
    def concsLoss(self, t, t_hat, type='ae'):
        t = t[:, :self.basisFSL.n_metabs]
        t_hat = t_hat[:, :self.basisFSL.n_metabs]

        if type == 'ae':  # absolute error
            return np.abs(t - t_hat)
        else:
            raise ValueError('Unknown loss type... Please use one of the predefined!')


    #*************************#
    #   optimal referencing   #
    #*************************#
    def optimalReference(self, t, t_hat):
        w = np.ones(t.shape[0])
        for i in range(t.shape[0]):
            def err(w):
                w = np.clip(w, 0, None)
                return np.abs(t[i] - w * t_hat[i]).mean()
            w[i] = minimize(err, w[i], bounds=[(0, None)]).x
        return w[..., np.newaxis]


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x_ref=None, frac=None, x0=None):
        theta = self.fsl_minimize(x, x_ref, frac, x0)
        return theta


    #*********************#
    #   FSL-MRS fitting   #
    #*********************#
    def fsl_minimize(self, x, x_ref=None, frac=None, x0=None):
        thetas, uncs = [], []
        x = x[:, 0] + 1j * x[:, 1]
        fids = np.fft.ifft(x, axis=-1)   # to time domain
        if self.conj:
            fids = np.conjugate(fids)   # conjugate if necessary
            if x_ref is not None: x_ref = np.conjugate(x_ref)

        # multi threading
        if self.multiprocessing:
            tasks = [(fid, self.basisFSL, self.method, x_ref, frac, x0, i) for i, fid in enumerate(fids)]
            with multiprocessing.Pool(None) as pool:
                thetas, uncs = zip(*pool.starmap(self.fsl_forward, tasks))

        else:  # loop
            for i, fid in enumerate(fids):
                theta, unc = self.fsl_forward(fid, self.basisFSL, self.method, x_ref, frac, x0, i)
                thetas.append(theta)
                uncs.append(unc)

        if self.include_params:
            return np.array(thetas), np.array(uncs)
        else:
            return np.array(thetas)[:, :self.basisFSL.n_metabs], \
                   np.array(uncs)[:, :self.basisFSL.n_metabs]


    #**************************#
    #   FSL-MRS optimization   #
    #**************************#
    def fsl_forward(self, fid, basis, method, ref=None, frac=None, x0=None, idx=0):
        specFSL = MRS(FID=fid,
                      basis=basis,
                      H2O=ref[idx] if not ref is None and ref.shape[0] > idx else ref,
                      cf=basis.cf,
                      bw=basis.original_bw,
                      nucleus=self.nucleus)
        specFSL.processForFitting()

        x0 = x0[idx] if not x0 is None and x0.shape[0] > idx else x0
        opt = fitting.fit_FSLModel(specFSL, method=method, x0=x0, ppmlim=self.ppmlim)

        if ref is not None:   # water scaling
            # TODO: the TE/TR should be inferred or otherwise passed
            try:
                assert self.TE is not None and self.TR is not None,\
                    'TE and TR must be provided for water scaling!'
                q_info = quantify.QuantificationInfo(self.TE, self.TR, specFSL.names,
                                                    specFSL.centralFrequency / 1E6)

                if frac is not None: q_info.set_fractions(frac[idx])

                opt.calculateConcScaling(specFSL, quant_info=q_info)
                concs = opt.getConc(scaling='molarity')
                uncs = opt.getUncertainties(type='molarity')
            except:
                print('Quantification Error! Skipping water referencing...')
                concs = opt.params
                uncs = opt.crlb
        else:
            concs = opt.params
            uncs = opt.crlb

        if self.save_path != '':   # save results
            if not os.path.exists(self.save_path): os.makedirs(self.save_path)
            opt.to_file(f'{self.save_path}/summary{idx}.csv', what='summary')
            opt.to_file(f'{self.save_path}/concs{idx}.csv', what='concentrations')
            opt.plot(specFSL, out=f'{self.save_path}/fit{idx}.png')
            pickle.dump(opt, open(f'{self.save_path}/opt{idx}.pkl', 'wb'))

            from fsl_mrs.utils import plotting
            plotting.plotly_fit(specFSL, opt).write_html(f'{self.save_path}/residuals{idx}.html')

        if self.unc == 'perc':   # uncertainty estimation
            params = opt.params
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_SD = np.sqrt(opt.crlb[:len(params)]) / params * 100
                perc_SD[perc_SD > 999] = 999
                perc_SD[np.isnan(perc_SD)] = 999
            return concs, perc_SD
        elif self.unc == 'crlb':
            return concs, uncs
        else: raise ValueError('Invalid uncertainty type!')



#*************#
#   testing   #
#*************#
if __name__ == '__main__':
    import os
    import pandas as pd

    config = {
        'path2basis': '../Data/basisset_JMRUI/',
        'path2concs': '../Data/ground_truth/',
        'path2data': '../Data/datasets_JMRUI_WS/',
        'path2water': '../Data/datasets_JMRUI_nWS/',

        'test_size': 2,  # number of test samples

        'method': 'Newton',   # 'MH' or 'Newton'

        'TE': 30 / 1e3,   # needed for water scaling
        'TR': 4,
    }

    fsl = FrameworkFSL(config['path2basis'], method=config['method'], TE=config['TE'], TR=config['TR'])

    def load_EXCEL_conc(path2conc):
        # Load a list of concentrations from an EXCEL file, specifically the ground
        # truth concentrations of the ISMRM 2016 fitting challenge.

        truth = {'Ace': 0.0}  # initialize, Ace is only partially present

        file = pd.read_excel(path2conc, header=17)
        for i, met in enumerate(file['Metabolites']):
            if not isinstance(met, str): break
            truth[met] = file['concentration'][i]

        truth['Mac'] = truth.pop('MMBL')  # rename key MMBL to Mac
        return dict(sorted(truth.items()))

    # load ground truth
    files = os.listdir(config['path2concs'])[:config['test_size']]
    concs = [load_EXCEL_conc(config['path2concs'] + file) for file in files]   # load excel
    concs = [[c[met] for met in fsl.basisFSL._names] for c in concs]   # sort by basis names
    concs = np.array(concs)[:, :fsl.basisFSL.n_metabs]

    # load data
    files = os.listdir(config['path2data'])[:config['test_size']]
    data = np.array([mrs_io.read_FID(config['path2data'] + file).mrs().FID for file in files])

    if 'path2water' in config:
        files = os.listdir(config['path2water'])[:config['test_size']]
        water = np.array([mrs_io.read_FID(config['path2water'] + file).mrs().FID for file in files])
    else:
        water = None

    # to frequency domain (stack real and imaginary part) <- this is my convention
    data = np.fft.fft(data, axis=-1)
    data = np.stack((data.real, data.imag), axis=1)

    # fit
    thetas, uncs = fsl(data, water)  # data in freq. domain, shape of (batch_dim, 2, sample_points)
                                           # water in time domain, shape of (batch_dim, sample_points)
    # loss
    if 'path2water' not in config: thetas = fsl.optimalReference(concs, thetas) * thetas
    loss = fsl.concsLoss(concs, thetas, type='ae')
    print('MAE:', loss.mean())


