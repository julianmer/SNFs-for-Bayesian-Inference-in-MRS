####################################################################################################
#                                          flowModels.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/24                                                                                #
#                                                                                                  #
# Purpose: Definitions of various VAE and normalizing flow models.                                 #
#          Adapted from: https://github.com/riannevdberg/sylvester-flows.                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn

# own
from simulation.simulation import extract_prior_params


#********************#
#   get activation   #
#********************#
def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


#**************************************************************************************************#
#                                      Masked Linear Layer (MLL)                                   #
#**************************************************************************************************#
#                                                                                                  #
# A masked linear layer.                                                                           #
#                                                                                                  #
#**************************************************************************************************#
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, diagonal_zeros=False, bias=True, device=None):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        weight = torch.FloatTensor(in_features, out_features)

        self.weight = nn.parameter.Parameter(weight).to(device)
        if bias:
            bias = torch.FloatTensor(out_features)
            self.bias = nn.parameter.Parameter(bias).to(device)
        else:
            self.register_parameter('bias', None)
        mask = self.build_mask()
        self.mask = torch.autograd.Variable(mask, requires_grad=False).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0

        mask = torch.ones((n_in, n_out), dtype=torch.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k:, i:i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask.to(x.device) * self.weight)
        if self.bias is not None: return output.add(self.bias.expand_as(output))
        else: return output

    def __repr__(self):
        if self.bias is not None: bias = True
        else: bias = False
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', diagonal_zeros=' \
            + str(self.diagonal_zeros) + ', bias=' \
            + str(bias) + ')'


#**************************************************************************************************#
#                                 Inverse Autoregressive Flow (IAF)                                #
#**************************************************************************************************#
#                                                                                                  #
# An implementation of the Inverse Autoregressive Flow (IAF) as described in the paper:            #
# "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, et al. #
#                                                                                                  #
#**************************************************************************************************#
class IAF(nn.Module):
    def __init__(self, latent_dim, num_flows, hidden_dim=64, num_hidden=2,
                 forget_bias=1.0, activation='relu'):
        super(IAF, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        self.forget_bias = forget_bias

        # reordering
        flip_idx = torch.arange(self.latent_dim - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # flows
        self.flows = nn.ModuleList([
            nn.ModuleDict({
                'z_feats': nn.Sequential(
                    MaskedLinear(latent_dim, hidden_dim),
                    nn.ReLU()
                ),
                'zh_feats': nn.Sequential(
                    *[nn.Sequential(
                        MaskedLinear(hidden_dim, hidden_dim),
                        nn.ReLU()
                    ) for _ in range(num_hidden)]
                ),
                'linear_mean': MaskedLinear(hidden_dim, latent_dim, diagonal_zeros=True),
                'linear_std': MaskedLinear(hidden_dim, latent_dim, diagonal_zeros=True)
            })
            for _ in range(num_flows)
        ])

    def forward(self, z, h_context):
        logdets = 0.
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0:
                z = z[:, self.flip_idx]
            h = flow['z_feats'](z) + h_context
            h = flow['zh_feats'](h)
            mean = flow['linear_mean'](h)
            gate = torch.sigmoid(flow['linear_std'](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(torch.log(gate), dim=1)
        return z, logdets



#**************************************************************************************************#
#                                          Sylvester Flow                                          #
#**************************************************************************************************#
#                                                                                                  #
# An implementation of the Sylvester normalizing flow.                                             #
#                                                                                                  #
#**************************************************************************************************#
class SylvesterFlow(nn.Module):
    def __init__(self, num_ortho_vecs):
        super(SylvesterFlow, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', torch.autograd.Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        # amortized flow parameters
        zk = zk.unsqueeze(1)

        # save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # compute log|det J|
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj: log_det_j = log_diag_j.sum(-1)
        else: log_det_j = log_diag_j
        return z, log_det_j


#**************************************************************************************************#
#                                          Base VAE Class                                          #
#**************************************************************************************************#
#                                                                                                  #
# The base class for a variational auto-encoder (VAE).                                             #
#                                                                                                  #
#**************************************************************************************************#
class BaseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder='default', decoder='default',
                 arch='mlp', dropout=0.0, width=64, depth=3, activation='relu', device=None,
                 **kwargs):
        super(BaseVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # prior
        dataSynth = kwargs['dataSynth']
        self.p_mu, self.p_var = extract_prior_params(dataSynth.basisObj,
                                                     dataSynth.params,
                                                     dataSynth.concs)
        self.p_mu, self.p_var = torch.tensor(self.p_mu), torch.tensor(self.p_var)
        self.p_var = self.p_var.clamp(min=1e-8)

        self.p_min, self.p_max = extract_prior_params(dataSynth.basisObj,
                                                      dataSynth.params,
                                                      dataSynth.concs,
                                                      mode='unif')
        self.p_min, self.p_max = torch.tensor(self.p_min), torch.tensor(self.p_max)

        # encoder
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder(arch, dropout,
                                                                       width, depth, activation)
        # decoder
        if self.decoder == 'model_based':
            assert 'sigModel' in kwargs, "sigModel must be provided for model-based decoding."
            self.sigModel = kwargs['sigModel']
        else:
            self.p_x_nn, self.p_x_mean = self.create_decoder(arch, dropout,
                                                             width, depth, activation)

        # log-det-jacobian term
        self.log_det_j = torch.zeros(1)

    def create_encoder(self, arch, dropout, width, depth, activation):
        if arch == 'mlp':
            input_dim = eval('*'.join(map(str, self.input_dim)))

            q_z_nn = [nn.Flatten(),
                      nn.BatchNorm1d(input_dim),
                      nn.Linear(input_dim, width),
                      get_activation(activation),
                      nn.Dropout(dropout)]

            for i in range(1, depth):
                q_z_nn.extend([nn.Linear(width // (2 ** (i - 1)), width // (2 ** i)),
                               get_activation(activation),
                               nn.Dropout(dropout)])

            q_z_nn.extend([nn.Linear(width // (2 ** (depth - 1)), 2 * self.latent_dim)])
            q_z_mean = nn.Linear(2 * self.latent_dim, self.latent_dim)
            q_z_var = [nn.Linear(2 * self.latent_dim, self.latent_dim),
                       # nn.Sigmoid(),
                       ]

            return nn.Sequential(*q_z_nn), q_z_mean, nn.Sequential(*q_z_var)

        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def create_decoder(self, arch, dropout, width, depth, activation):
        if arch == 'mlp':
            input_dim = eval('*'.join(map(str, self.input_dim)))

            p_x_nn = [nn.Linear(self.latent_dim, width // (2 ** (depth - 1))),
                      get_activation(activation),
                      nn.Dropout(dropout)]

            for i in reversed(range(1, depth)):
                p_x_nn.extend([nn.Linear(width // (2 ** i), width // (2 ** (i - 1))),
                               get_activation(activation),
                               nn.Dropout(dropout)])

            p_x_nn.extend([nn.Linear(width, input_dim)])

            p_x_mean = [nn.Linear(input_dim, input_dim),
                        nn.Unflatten(1, self.input_dim)]

            return nn.Sequential(*p_x_nn), nn.Sequential(*p_x_mean)

        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def encode(self, x):
        if self.encoder == 'default':
            h = self.q_z_nn(x)
            mu = self.q_z_mean(h)
            logvar = self.q_z_var(h)
            return mu, logvar

        else:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def apply_prior(self, z):
        return self.p_mu.to(z.device) + self.p_var.sqrt().to(z.device) * z

    def decode(self, z):
        if self.decoder == 'default':
            h = self.p_x_nn(z)
            x_mean = self.p_x_mean(h)
            return x_mean.double()

        elif self.decoder == 'model_based':
            z = self.apply_prior(z)

            # # make sure z is in bounds
            # z = z.clamp(min=self.p_min.to(z.device), max=self.p_max.to(z.device))
            n = self.sigModel.basis.shape[1]
            clamped = z[:, :n].clamp(0, torch.inf)
            rest = z[:, n:]
            z = torch.cat([clamped, rest], dim=1)

            spec = self.sigModel.forward(z)
            spec = torch.stack((spec.real, spec.imag), dim=1)
            spec = spec[..., self.sigModel.first:self.sigModel.last]
            return spec

        else:
            raise ValueError(f"Unsupported decoder: {self.decoder}")

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)   # sample z
        x_mean = self.decode(z)
        return x_mean, mu, logvar, self.log_det_j.to(x.device), z, z


#**************************************************************************************************#
#                                           IAF VAE Class                                          #
#**************************************************************************************************#
#                                                                                                  #
# A variational auto-encoder (VAE) with an inverse autoregressive flow (IAF) as the prior.         #
#                                                                                                  #
#**************************************************************************************************#
class IAFVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, encoder='default', decoder='default',
                 arch='mlp', dropout=0.0, width=64, depth=3, activation='relu', num_flows=8,
                 hidden_dim=64, num_hidden=2, forget_bias=1.0, **kwargs):
        super(IAFVAE, self).__init__(input_dim, latent_dim, encoder, decoder, arch, dropout,
                                     width, depth, activation, **kwargs)
        self.flow = IAF(latent_dim, num_flows, hidden_dim, num_hidden, forget_bias, activation)
        self.q_z_nn_output_dim = 2 * self.latent_dim
        self.h_context = nn.Linear(self.q_z_nn_output_dim, hidden_dim)

    def encode(self, x):
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        h_context = self.h_context(h)
        return mean_z, var_z, h_context

    def forward(self, x):
        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        zk, self.log_det_j = self.flow(z0, h_context)

        # decode
        x_mean = self.decode(zk)
        return x_mean, z_mu, z_var, self.log_det_j, z0, zk


#**************************************************************************************************#
#                                       Sylvester VAE Class                                        #
#**************************************************************************************************#
#                                                                                                  #
# A variational auto-encoder (VAE) with Sylvester normalizing flows.                               #
#                                                                                                  #
#**************************************************************************************************#
class SylvesterVAE(BaseVAE):
    def __init__(self, input_dim, latent_dim, encoder='default', decoder='default',
                 arch='mlp', dropout=0.0, width=64, depth=3, activation='relu', num_flows=8,
                 num_ortho_vecs=8, num_householder=8, **kwargs):
        super(SylvesterVAE, self).__init__(input_dim, latent_dim, encoder, decoder, arch,
                                           dropout, width, depth, activation, **kwargs)

        self.q_z_nn_output_dim = 2 * self.latent_dim

        # flow parameters
        flow = SylvesterFlow
        self.num_flows = num_flows
        self.num_householder = num_householder
        assert self.num_householder > 0

        identity = torch.eye(self.latent_dim, self.latent_dim).unsqueeze(0)

        # put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', torch.autograd.Variable(identity))
        self._eye.requires_grad = False

        # masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.latent_dim, self.latent_dim), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.latent_dim).long()

        self.register_buffer('triu_mask', torch.autograd.Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # amortized flow parameters
        # diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.latent_dim * self.latent_dim)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.latent_dim),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.latent_dim),
            self.diag_activation
        )

        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.latent_dim * self.num_householder)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.latent_dim)

        # normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.latent_dim)
            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        # reshape to shape (num_flows * batch_size * num_householder, z_size)
        q = q.view(-1, self.latent_dim)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)   # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # calculate Householder matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        # reshaping: first dimension is batch_size * num_flows
        amat = amat.view(-1, self.num_householder, self.latent_dim, self.latent_dim)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.num_flows, self.latent_dim, self.latent_dim)
        amat = amat.transpose(0, 1)
        return amat

    def encode(self, x):
        batch_size = x.size(0)

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # amortized r1, r2, q, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.resize(batch_size, self.latent_dim, self.latent_dim, self.num_flows)
        diag1 = diag1.resize(batch_size, self.latent_dim, self.num_flows)
        diag2 = diag2.resize(batch_size, self.latent_dim, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(h)
        b = self.amor_b(h)

        # resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.latent_dim, self.num_flows)
        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        self.log_det_j = 0.
        batch_size = x.size(0)

        z_mu, z_var, r1, r2, q, b = self.encode(x)

        # orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k],
                                           q_k, b[:, :, :, k], sum_ldj=True)

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_mean = self.decode(z[-1])
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]