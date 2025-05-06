####################################################################################################
#                                          simulation.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 15/12/22                                                                                #
#                                                                                                  #
# Purpose: Simulate a corpus of MRS data.                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

# own
from simulation.simulationDefs import normParams, normConcs


#*********************#
#   draw parameters   #
#*********************#
def simulateParam(basis, batch, params=normParams, concs=normConcs):
    """
    Function to simulate SVS parameters.

    @param basis -- The basis set of metabolites to simulate as (FSL) MRS object.
    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.
    @param concs -- The concentration ranges of the metabolites in form of a dictionary,
                    if not given standard concentrations will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    # get metabolite concentrations
    randomConc = {}
    for name in basis.names:
        cName = name.split('.')[0]   # remove format ending (e.g. 'Ace.raw' -> 'Ace')

        #  draw randomly from range
        randomConc[name] = dist(concs[cName]['low_limit'], concs[cName]['up_limit'], batch)

    gamma = dist(params['broadening'][0][0], params['broadening'][1][0], batch)
    sigma = dist(params['broadening'][0][1], params['broadening'][1][1], batch)
    shifting = dist(params['shifting'][0], params['shifting'][1], batch)
    phi0 = dist(params['phi0'][0], params['phi0'][1], batch)
    phi1 = dist(params['phi1'][0], params['phi1'][1], batch)

    theta = np.array(list(randomConc.values()))
    theta = np.concatenate((theta, gamma[np.newaxis, :]))
    theta = np.concatenate((theta, sigma[np.newaxis, :]))
    theta = np.concatenate((theta, shifting[np.newaxis, :]))
    theta = np.concatenate((theta, phi0[np.newaxis, :]))
    theta = np.concatenate((theta, phi1[np.newaxis, :]))

    for i in range(len(params['baseline'][0])):
        theta = np.concatenate((theta, dist(params['baseline'][0][i],
                                            params['baseline'][1][i], batch)[np.newaxis, :]))

    if 'noise' in params:
        noise = np.random.normal(params['noise'][0], params['noise'][1], (batch, basis.fids.shape[0])) + \
                1j * np.random.normal(params['noise'][0], params['noise'][1], (batch, basis.fids.shape[0]))
    elif 'noise_mean' in params and 'noise_std' in params:
        mean = dist(params['noise_mean'][0], params['noise_mean'][1], (batch, 1))
        std = dist(params['noise_std'][0], params['noise_std'][1], (batch, 1))
        noise = np.random.normal(mean, std, (batch, basis.fids.shape[0])) + \
                1j * np.random.normal(mean, std, (batch, basis.fids.shape[0]))
    else:
        noise = np.zeros((batch, basis.fids.shape[0]))

    return np.swapaxes(theta, 0, 1), noise


#*************************************#
#   draw parameters for random walk   #
#*************************************#
def simulateRW(batch, params=normParams):
    """
    Function to simulate random walk parameters.

    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    scale = dist(params['scale'][0], params['scale'][1], batch)
    smooth = dist(params['smooth'][0], params['smooth'][1], batch)
    lowLimit = dist(params['limits'][0][0], params['limits'][0][1], batch)
    highLimit = dist(params['limits'][1][0], params['limits'][1][1], batch)
    return scale, smooth, lowLimit, highLimit


#**************************************#
#   draw parameters for random peaks   #
#**************************************#
def simulatePeaks(batch, params=normParams):
    """
    Function to simulate peaks parameters.

    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    amps = dist(params['peakAmp'][0], params['peakAmp'][1], batch)[:, np.newaxis]
    widths = dist(params['peakWidth'][0], params['peakWidth'][1], batch)[:, np.newaxis]
    phases = dist(params['peakPhase'][0], params['peakPhase'][1], batch)[:, np.newaxis]
    return amps, widths, phases


#***********************************#
#   extract priors for parameters   #
#***********************************#
def extract_prior_params(basis, params=normParams, concs=normConcs, mode='normal'):
    """
    Function to extract the priors for the parameters.

    @param basis -- The basis set of metabolites to simulate as (FSL) MRS object.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.
    @param concs -- The concentration ranges of the metabolites in form of a dictionary,
                    if not given standard concentrations will be used.
    @param mode -- The mode of the distribution ('normal' or 'unif').

    @returns -- The mean and variance (or min and max) of the parameters.
    """
    if mode == 'normal':
        mu = np.zeros(len(basis.names) + 11)
        var = np.zeros(len(basis.names) + 11)
    elif mode == 'unif':
        mins = np.zeros(len(basis.names) + 11)
        maxs = np.zeros(len(basis.names) + 11)
    else:
        raise ValueError('Unknown mode.')

    if params['dist'] == 'unif':
        for i, name in enumerate(basis.names):
            low, up = concs[name]['low_limit'], concs[name]['up_limit']
            if mode == 'normal':
                mu[i] = (low + up) / 2
                var[i] = (up - low) ** 2 / 12
            elif mode == 'unif':
                mins[i] = low
                maxs[i] = up

        for i, (low, up) in enumerate(zip(params['broadening'][0], params['broadening'][1])):
            if mode == 'normal':
                mu[i + len(basis.names)] = (low + up) / 2
                var[i + len(basis.names)] = (up - low) ** 2 / 12
            elif mode == 'unif':
                mins[i + len(basis.names)] = low
                maxs[i + len(basis.names)] = up

        for i, name in enumerate(['shifting', 'phi0', 'phi1']):
            low, up = params[name]
            if mode == 'normal':
                mu[i + 2 + len(basis.names)] = (low + up) / 2
                var[i + 2 + len(basis.names)] = (up - low) ** 2 / 12
            elif mode == 'unif':
                mins[i + 2 + len(basis.names)] = low
                maxs[i + 2 + len(basis.names)] = up

        for i, (low, up) in enumerate(zip(params['baseline'][0], params['baseline'][1])):
            if mode == 'normal':
                mu[i + 5 + len(basis.names)] = (low + up) / 2
                var[i + 5 + len(basis.names)] = (up - low) ** 2 / 12
            elif mode == 'unif':
                mins[i + 5 + len(basis.names)] = low
                maxs[i + 5 + len(basis.names)] = up

    elif params['dist'] == 'normal':
        for i, name in enumerate(basis.names):
            mean, std = concs[name]['low_limit'], concs[name]['up_limit']
            if mode == 'normal':
                mu[i] = mean
                var[i] = std ** 2
            elif mode == 'unif':
                mins[i] = np.clip(mean - 3 * std, 0, None)
                maxs[i] = mean + 3 * std

        for i, (mean, std) in enumerate(zip(params['broadening'][0], params['broadening'][1])):
            if mode == 'normal':
                mu[i + len(basis.names)] = mean
                var[i + len(basis.names)] = std ** 2
            elif mode == 'unif':
                mins[i + len(basis.names)] = np.clip(mean - 3 * std, 0, None)
                maxs[i + len(basis.names)] = mean + 3 * std

        for i, name in enumerate(['shifting', 'phi0', 'phi1']):
            mean, std = params[name]
            if mode == 'normal':
                mu[i + 2 + len(basis.names)] = mean
                var[i + 2 + len(basis.names)] = std ** 2
            elif mode == 'unif':
                mins[i + 2 + len(basis.names)] = mean - 3 * std
                maxs[i + 2 + len(basis.names)] = mean + 3 * std

        for i, (mean, std) in enumerate(zip(params['baseline'][0], params['baseline'][1])):
            if mode == 'normal':
                mu[i + 5 + len(basis.names)] = mean
                var[i + 5 + len(basis.names)] = std ** 2
            elif mode == 'unif':
                mins[i + 5 + len(basis.names)] = mean - 3 * std
                maxs[i + 5 + len(basis.names)] = mean + 3 * std

    if mode == 'normal': return mu, var
    elif mode == 'unif': return mins, maxs