####################################################################################################
#                                           sigModels.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/07/22                                                                                #
#                                                                                                  #
# Purpose: Definitions of various MRS signal models.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

from fsl_mrs.core import MRS
from fsl_mrs.models import getModelFunctions, getModelJac
from fsl_mrs.models.model_voigt import init
from fsl_mrs.utils.misc import calculate_lap_cov

from scipy.linalg import lstsq as sp_lstsq
from scipy.optimize import minimize



#**************************************************************************************************#
#                                          Class SigModel                                          #
#**************************************************************************************************#
#                                                                                                  #
# The base class for the MRS signal models. Defines the necessary attributes and methods a signal  #
# model should implement.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class SigModel():

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis, baseline, order, first, last, t, f):
        self.basis = basis
        self.first, self.last = first, last

        if not baseline:
            self.baseline = \
            torch.from_numpy(self.baseline_init(order, first, last))
        else:
            self.baseline = baseline

        self.t = torch.from_numpy(t)
        self.f = torch.from_numpy(f)
        self.basis = torch.from_numpy(basis)


    #********************#
    #   parameter init   #
    #********************#
    def initParam(self, specs):
        pass


    #*******************#
    #   forward model   #
    #*******************#
    def forward(self, theta):
        pass


    #***************#
    #   regressor   #
    #***************#
    def regress_out(self, x, conf, keep_mean=True):
        """
        Linear deconfounding

        Ref: Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package.
        Magnetic Resonance in Medicine 2021;85:2950–2964 doi: https://doi.org/10.1002/mrm.28630.
        """
        if isinstance(conf, list):
            confa = np.squeeze(np.asarray(conf)).T
        else:
            confa = conf
        if keep_mean:
            m = np.mean(x, axis=0)
        else:
            m = 0
        return x - confa @ (np.linalg.pinv(confa) @ x) + m


    #*******************#
    #   baseline init   #
    #*******************#
    def baseline_init(self, order, first, last):
        x = np.zeros(self.basis.shape[0], complex)
        x[first:last] = np.linspace(-1, 1, last - first)
        B = []
        for i in range(order + 1):
            regressor = x ** i
            if i > 0:
                regressor = self.regress_out(regressor, B, keep_mean=False)

            B.append(regressor.flatten())
            B.append(1j * regressor.flatten())

        B = np.asarray(B).T
        tmp = B.copy()
        B = 0 * B
        B[first:last, :] = tmp[first:last, :].copy()
        return B



#**************************************************************************************************#
#                                         Class VoigtModel                                         #
#**************************************************************************************************#
#                                                                                                  #
# Implements a signal model as mentioned in [1] (a Voigt signal model).                            #
#                                                                                                  #
# [1] Clarke, W.T., Stagg, C.J., and Jbabdi, S. (2020). FSL-MRS: An end-to-end spectroscopy        #
#     analysis package. Magnetic Resonance in Medicine, 85, 2950 - 2964.                           #
#                                                                                                  #
#**************************************************************************************************#
class VoigtModel(SigModel):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis, first, last, t , f, baseline=None, order=2):
        """
        Main init for the VoigtModel class.

        @param basis -- The basis set of metabolites.
        @param baseline -- The baseline used to fit the spectra.
        """
        super(VoigtModel, self).__init__(basis, baseline, order,
                                         first=first, last=last, t=t, f=f)

    #********************#
    #   parameter init   #
    #********************#
    def initParam(self, specs, mode='fsl', basisFSL=None):
        """
        Initializes the optimization parameters.

        @param specs -- The batch of specs to get initializations for.
        @param mode -- The initialization mode (default: 'fsl').
        @param basisFSL -- The basis set of metabolites as (FSL) MRS object (default: None).

        @returns -- The optimization parameters.
        """
        if mode.lower() == 'fsl':
            theta = np.zeros((specs.shape[0], self.basis.shape[1] + 11))
            for i, spec in enumerate(specs):
                specFSL = MRS(FID=np.fft.ifft(spec.cpu().numpy()),
                              basis=basisFSL,
                              cf=basisFSL.cf,
                              bw=basisFSL.original_bw)
                specFSL.processForFitting()
                theta[i, :] = init(specFSL,
                                   metab_groups=[0],
                                   baseline=self.baseline.detach().cpu().numpy(),
                                   ppmlim=[0.2, 4.2])
        elif mode.lower() == 'compute':
            theta = self.compute_init_params(specs)
        elif mode.lower() == 'random' or mode.lower() == 'rand':
            theta = np.zeros((specs.shape[0], self.basis.shape[1] + 11))
            theta[:, :self.basis.shape[1]] = np.random.rand(specs.shape[0], self.basis.shape[1])
            theta[:, self.basis.shape[1]:self.basis.shape[1] + 2] = np.ones((specs.shape[0], 2))
        elif mode.lower() == 'zero':
            theta = np.zeros((specs.shape[0], self.basis.shape[1] + 11))
        else:
            raise ValueError('Unknown initialization mode: ' + mode)
        return torch.Tensor(theta)


    #*******************#
    #   broaden basis   #
    #*******************#
    def modify_basis(self, gamma, sigma, eps, first=None, last=None):
        """
        Apply the Voigt lineshape to the time-domain basis and transform to frequency domain.

        @param gamma -- The Lorentzian blurring.
        @param sigma -- The Gaussian broadening.
        @param eps -- The frequency shift.
        @param first -- The first frequency index (default: instance attribute).
        @param last -- The last frequency index (default: instance attribute).

        @returns -- The modified basis in the frequency domain.
        """
        if first is None: first = self.first
        if last is None: last = self.last

        if len(gamma.shape) == 0: gamma = torch.tensor(gamma).unsqueeze(0)
        if len(sigma.shape) == 0: sigma = torch.tensor(sigma).unsqueeze(0)
        if len(eps.shape) == 0: eps = torch.tensor(eps).unsqueeze(0)
        gamma = gamma.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        eps = eps.unsqueeze(1)
        t = self.t.unsqueeze(0)
        exp_term = torch.exp(- (gamma + sigma ** 2 * t) * t - 1j * eps * t)
        bs = self.basis.unsqueeze(0) * exp_term.unsqueeze(-1)
        bs_spec = torch.fft.fft(bs, dim=-2)
        bs_spec = bs_spec[:, first:last, :]
        return torch.cat((bs_spec.real, bs_spec.imag), dim=1).squeeze(0)


    #********************#
    #   compute params   #
    #********************#
    def compute_init_params(self, data, first=None, last=None):
        """
        Estimate parameters by fitting a modified basis to the measured data.

        Parameters:
        @param data (torch.Tensor): The measured data with shape (batch, n_freq).
        @param first (int, optional): The first frequency index. Defaults to instance attribute.
        @param last (int, optional): The last frequency index. Defaults to instance attribute.

        Returns:
        @returns tuple: Estimated parameters cat(con_init, gamma, sigma, eps, phi0, phi1, b_params) for each batch.
        """
        if first is None: first = self.first
        if last is None: last = self.last

        batch_size = data.size(0)
        n_basis = self.basis.size(1)
        nb = self.baseline.size(1)

        # prepare variables
        baseline = self.baseline[first:last]
        baseline = torch.cat((torch.real(baseline), torch.imag(baseline)), dim=0)
        baseline = baseline.repeat(batch_size, 1, 1)

        data = torch.cat((data[:, first:last].real, data[:, first:last].imag), dim=1)

        # initialize parameters
        con_init = torch.zeros(batch_size, n_basis, requires_grad=True)
        gamma_vals = torch.zeros(batch_size, requires_grad=True)
        sigma_vals = torch.zeros(batch_size, requires_grad=True)
        eps_vals = torch.zeros(batch_size, requires_grad=True)
        phi0_vals = torch.zeros(batch_size, requires_grad=True)
        phi1_vals = torch.zeros(batch_size, requires_grad=True)
        b_params = torch.zeros(batch_size, nb, requires_grad=True)
        for i in range(batch_size):
            # loss function
            def loss(p, baseline, data, first=None, last=None):
                gamma, sigma, eps = np.exp(p[0]), np.exp(p[1]), p[2]
                basis = self.modify_basis(gamma, sigma, eps, first, last)
                desmat = np.concatenate((basis, baseline), axis=1)
                # pinv = np.linalg.pinv(desmat)
                # beta = np.real(pinv @ y)
                # beta = np.linalg.lstsq(desmat, y)[0]
                beta = sp_lstsq(desmat, data, lapack_driver='gelsy', check_finite=False)[0]
                # project onto >0 concentration
                beta[:n_basis] = np.clip(beta[:n_basis], 0, None)
                pred = np.matmul(desmat, beta)
                val = np.mean(np.abs(pred - data.numpy()) ** 2)
                return val

            # optimize
            bounds = (
                (None, np.log(100)),
                (None, np.log(100)),
                (-300, 300),
            )
            res = minimize(loss, np.array([np.log(1), np.log(1), 0]),
                           args=(baseline[i], data[i], first, last),
                           bounds=bounds)
            gamma_vals[i] = np.exp(res.x[0])
            sigma_vals[i] = np.exp(res.x[1])
            eps_vals[i] = res.x[2]

            # initialize concentrations
            basis = self.modify_basis(gamma_vals[i], sigma_vals[i], eps_vals[i], first, last)
            desmat = np.concatenate((basis, baseline[i]), axis=1)
            # params = sp_lstsq(desmat, data[i], lapack_driver='gelsy', check_finite=False)[0]
            params = np.real(np.linalg.pinv(desmat) @ data[i].numpy())
            con_init[i] = torch.clamp(torch.Tensor(params[:n_basis]), 0, None)
            phi0_vals[i] = 0
            phi1_vals[i] = 0
            b_params[i] = torch.Tensor(params[n_basis:])

        return torch.cat((con_init, gamma_vals.unsqueeze(1), sigma_vals.unsqueeze(1),
                          eps_vals.unsqueeze(1), phi0_vals.unsqueeze(1), phi1_vals.unsqueeze(1),
                          b_params), dim=1)


    #*******************#
    #   forward model   #
    #*******************#
    def forward(self, theta, sumOut=True, baselineOut=False, phase1=True):
        """
        The (forward) signal model.

        @param theta -- The optimization parameters.
        @param sumOut -- Whether to sum over the metabolites (default: True).
        @param baselineOut -- Whether to return the baseline (default: False).
        @param phase1 -- Whether to include the first-order phase (default: True).

        @returns -- The forward model function.
        """
        self.t , self.f = self.t.to(theta.device), self.f.to(theta.device)
        self.basis = self.basis.to(theta.device)

        n = self.basis.shape[1]
        g = 1

        con = theta[:, :n]  # concentrations
        gamma = theta[:, n:n + g]  # lorentzian blurring
        sigma = theta[:, n + g:n + 2 * g]  # gaussian broadening
        eps = theta[:, n + 2 * g:n + 3 * g]  # frequency shift
        phi0 = theta[:, n + 3 * g]  # global phase shift
        phi1 = theta[:, n + 3 * g + 1]  # global phase ramp
        b = theta[:, n + 3 * g + 2:]  # baseline params

        # compute m(t) * exp(- (1j * eps + gamma + sigma ** 2 * t) * t)
        lin = torch.exp(- (1j * eps + gamma + (sigma ** 2) * self.t) * self.t)
        ls = lin[..., None] * self.basis
        S = torch.fft.fft(ls, dim=1)

        # compute exp(-1j * (phi0 + phi1 * nu)) * con * S(nu)
        if phase1: ex = torch.exp(-1j * (phi0[..., None] + phi1[..., None] * self.f))
        else: ex = torch.exp(-1j * phi0[..., None])
        fd = ex[:, :, None] * con[:, None, :] * S

        # add baseline
        if self.baseline is not None:
            self.baseline = self.baseline.to(theta.device)

            # compute baseline
            if len(self.baseline.shape) > 2:
                ba = torch.einsum("ij, ikj -> ik", b.cfloat(), self.baseline)
            else:
                ba = torch.einsum("ij, kj -> ik", b.cdouble(), self.baseline)

        if sumOut: fd = fd.sum(-1) + ba
        if baselineOut: return fd, ba
        return fd


    #********************#
    #   error function   #
    #********************#

    def err_func(self, theta, data, first=None, last=None):
        """
        Compute the sum–of–squared error (SSE) over the frequency range [first:last].

        @param theta -- The optimization parameters.
        @param data -- The measured data.
        @param first -- The first frequency index.
        @param last -- The last frequency index.

        @returns -- The SSE.
        """
        if first is None: first = self.first
        if last is None: last = self.last
        fd = self.forward(theta, sumOut=True, baselineOut=False)
        E = data[:, first:last] - fd[:, first:last]
        return torch.real(torch.sum(E * torch.conj(E)))


    #********************#
    #   gradient model   #
    #********************#
    def gradient(self, theta, data, first=None, last=None, phase1=True):
        """
        Manual gradient of the error function (without autograd).

        @param theta -- The optimization parameters.
        @param data -- The measured data.
        @param first -- The first frequency index.
        @param last -- The last frequency index.
        @param phase1 -- Whether to include the first-order phase (default: True).

        @returns -- Returns a tensor of shape (B, P) representing the gradient.
        """
        n = self.basis.shape[1]
        if first is None: first = self.first
        if last is None: last = self.last

        # unpack theta using our x2p helper:
        con = theta[:, :n]  # (B, n)
        gamma = theta[:, n:n + 1]  # (B, 1)
        sigma = theta[:, n + 1:n + 2]  # (B, 1)
        eps = theta[:, n + 2:n + 3]  # (B, 1)
        phi0 = theta[:, n + 3]  # (B,)
        phi1 = theta[:, n + 4]  # (B,)
        b_param = theta[:, n + 3 + 2:]  # (B, nb)

        t = self.t  # (n_time,)
        f = self.f  # (n_freq,)
        t_exp = t.unsqueeze(0)  # (1, n_time)
        lin = torch.exp(- (1j * eps + gamma + sigma ** 2 * t_exp) * t_exp)  # (B, n_time)
        if self.basis.dim() == 2:
            basis_ = self.basis.unsqueeze(0)
        else:
            basis_ = self.basis
        ls = lin.unsqueeze(-1) * basis_
        S = torch.fft.fft(ls, dim=1)  # (B, n_freq, n)

        if phase1:
            ex = torch.exp(-1j * (phi0.unsqueeze(-1) + phi1.unsqueeze(-1) * f.unsqueeze(0)))  # (B, n_freq)
        else:
            ex = torch.exp(-1j * phi0.unsqueeze(-1))
        fd_components = ex.unsqueeze(-1) * con.unsqueeze(1) * S  # (B, n_freq, n)
        signal = fd_components.sum(-1)  # (B, n_freq)
        if self.baseline is not None:
            ba = torch.einsum("ij,kj->ik", b_param.cdouble(), self.baseline)  # (B, n_freq)
        else:
            ba = 0
        fd_calc = signal + ba  # (B, n_freq)
        E = data[:, first:last] - fd_calc[:, first:last]  # (B, freq_range)

        # compute partial derivatives manually:
        dfd_dcon = ex.unsqueeze(-1) * S  # (B, n_freq, n)
        dfd_dphi0 = -1j * (ex.unsqueeze(-1) * con.unsqueeze(1) * S).sum(-1)  # (B, n_freq)
        dfd_dphi1 = -1j * (f.unsqueeze(0).unsqueeze(-1) * ex.unsqueeze(-1) * con.unsqueeze(1) * S).sum(-1)  # (B, n_freq, n)
        dlin_dgamma = - t_exp * lin  # (B, n_time)
        dls_dgamma = dlin_dgamma.unsqueeze(-1) * basis_  # (B, n_time, n)
        dS_dgamma = torch.fft.fft(dls_dgamma, dim=1)  # (B, n_freq, n)
        dfd_dgamma = (ex.unsqueeze(-1) * con.unsqueeze(1) * dS_dgamma).sum(-1)  # (B, n_freq)
        dlin_dsigma = -2 * sigma * (t_exp ** 2) * lin  # (B, n_time)
        dls_dsigma = dlin_dsigma.unsqueeze(-1) * basis_  # (B, n_time, n)
        dS_dsigma = torch.fft.fft(dls_dsigma, dim=1)  # (B, n_freq, n)
        dfd_dsigma = (ex.unsqueeze(-1) * con.unsqueeze(1) * dS_dsigma).sum(-1)  # (B, n_freq)
        dlin_deps = -1j * t_exp * lin  # (B, n_time)
        dls_deps = dlin_deps.unsqueeze(-1) * basis_  # (B, n_time, n)
        dS_deps = torch.fft.fft(dls_deps, dim=1)  # (B, n_freq, n)
        dfd_deps = (ex.unsqueeze(-1) * con.unsqueeze(1) * dS_deps).sum(-1)  # (B, n_freq)
        if self.baseline is not None:
            dfd_db = self.baseline.unsqueeze(0).expand(theta.shape[0], -1, -1)  # (B, n_freq, nb)
        else:
            dfd_db = None

        # now sum contributions over frequency range [first:last]:
        grad_con = -2 * torch.real(torch.sum(E.unsqueeze(-1) * torch.conj(dfd_dcon[:, first:last, :]), dim=1))  # (B, n)
        grad_phi0 = -2 * torch.real(torch.sum(E * torch.conj(dfd_dphi0[:, first:last]), dim=1)).unsqueeze(1)  # (B, 1)
        grad_phi1 = -2 * torch.real(torch.sum(E * torch.conj(dfd_dphi1[:, first:last]), dim=1)).unsqueeze(1)  # (B, 1)
        grad_gamma = -2 * torch.real(torch.sum(E * torch.conj(dfd_dgamma[:, first:last]), dim=1)).unsqueeze(1)  # (B, 1)
        grad_sigma = -2 * torch.real(torch.sum(E * torch.conj(dfd_dsigma[:, first:last]), dim=1)).unsqueeze(1)  # (B, 1)
        grad_eps = -2 * torch.real(torch.sum(E * torch.conj(dfd_deps[:, first:last]), dim=1)).unsqueeze(1)  # (B, 1)
        if dfd_db is not None:
            grad_b = -2 * torch.real(
                torch.sum(E.unsqueeze(-1) * torch.conj(dfd_db[:, first:last, :]), dim=1))  # (B, nb)
        else:
            grad_b = None
        grad_list = [grad_con, grad_gamma, grad_sigma, grad_eps, grad_phi0, grad_phi1]
        if grad_b is not None:
            grad_list.append(grad_b)
        grad_total = torch.cat(grad_list, dim=1)
        return grad_total


    #**********************#
    #   CRLB computation   #
    #**********************#
    def crlb(self, theta, data, grad=None, sigma=None):
        """
        Computes the Cramer-Rao lower bound.

        @param theta -- The optimization parameters.
        @param data -- The data to compute the CRLB for.
        @param grad -- The gradient of the signal model (optional).
        @param sigma -- The noise std (optional).

        @returns -- The CRLB.
        """
        data = data[:, 0] + 1j * data[:, 1]

        if grad is None: grad = self.gradient(theta, data)
        if sigma is None:
            spec = self.forward(theta)
            if spec.shape[1] > data.shape[1]: spec = spec[:, self.first:self.last]
            sigma = torch.std(data - spec, dim=-1)[:, None, None]

        # compute the Fisher information matrix
        F0 = torch.diag(torch.ones(theta.shape[1]) * 1e-10)   # creates non-zeros on the diag
                                                              # (inversion can otherwise fail)
        F0 = F0.to(theta.device).unsqueeze(0)
        F = 1 / sigma **2 * torch.real(torch.permute(grad, [0, 2, 1]) @ torch.conj(grad))

        # compute the CRLB
        crlb = torch.sqrt(torch.linalg.inv(F + F0).diagonal(dim1=1, dim2=2))
        return crlb


    #********************************#
    #   CRLB computation using fsl   #
    #********************************#
    def crlb_fsl(self, theta, data, basis=None, grad=None, sigma=None):
        """
        Computes the Cramer-Rao lower bound.

        @param theta -- The optimization parameters.
        @param data -- The data to compute the CRLB for.
        @param basisFSL -- The basis set.
        @param grad -- The gradient of the signal model (optional).
        @param sigma -- The noise std (optional).

        @returns -- The CRLB.
        """
        data = data[:, 0] + 1j * data[:, 1]

        if basis is None: basis = self.basis.detach().cpu().numpy()

        _, _, forward, _, _ = getModelFunctions('voigt')
        jac = getModelJac('voigt')

        def forward_lim(theta):
            return forward(
                theta,
                self.f.detach().cpu().numpy(),
                self.t.detach().cpu().numpy(),
                basis,
                self.baseline.detach().cpu().numpy(),
                G=[0] * basis.shape[1],
                g=1,
            )[self.first:self.last]

        def jac_lim(theta):
            return jac(
                theta,
                self.f.unsqueeze(-1).detach().cpu().numpy(),
                self.t.unsqueeze(-1).detach().cpu().numpy(),
                basis,
                self.baseline.detach().cpu().numpy(),
                G=[0] * basis.shape[1],
                g=1, first=self.first, last=self.last)

        crlbs = np.zeros(theta.shape)
        for i in range(data.shape[0]):
            C = calculate_lap_cov(theta[i].detach().cpu().numpy(),
                                  forward_lim,
                                  data[i].detach().cpu().numpy(),
                                  jac_lim(theta[i].detach().cpu().numpy()).T)

            # scaling according to FSL-MRS
            crlbs[i] = np.sqrt(np.diag(C / 2))
        return torch.from_numpy(crlbs).to(theta.device)

