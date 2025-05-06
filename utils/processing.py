####################################################################################################
#                                           processing.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 21/06/22                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions for processing MRS data are defined here.                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

from fsl_mrs.utils.preproc.nifti_mrs_proc import update_processing_prov, DimensionsDoNotMatch

# from suspect.processing.denoising import sliding_gaussian


#*******************************#
#   process the basis spectra   #
#*******************************#
def processBasis(basis):
    """
        Processes the basis spectra.
        @param basis -- The basis spectra.
        @return -- The processed basis spectra.
    """
    # conjugate basis if necessary
    specBasis = np.abs(np.fft.fft(basis[:, 0]))
    if np.max(specBasis[:basis.shape[0] // 2]) > np.max(specBasis[basis.shape[0] // 2:]):
        basis = np.conjugate(basis)
    return basis # torch.from_numpy(basis).cfloat()


#*********************************#
#   process model input spectra   #
#*********************************#
def processSpectra(spectra, basis=None, conj=False):
    """
        Processes the input spectra.

        @param spectra -- The input spectra.
        @param basis -- The basis spectra.
        @param conj -- If True the spectra are conjugated (if necessary).

        @returns -- The processed spectra.
    """
    # conjugate spectra if necessary
    if conj and spectra[0, :spectra.shape[1] // 2].abs().max() > \
            spectra[0, spectra.shape[1] // 2:].abs().max():
        spectra = torch.conj(spectra)

    spectra = torch.stack((spectra.real, spectra.imag), dim=1)
    return spectra


#*********************************#
#   phase correction of spectra   #
#*********************************#
def phaseCorrection(spectra):
    """
        Phase correction of the spectra.

        @param spectra -- The spectra.

        @returns -- The phase corrected spectra. Aligned to the maximum peak.
    """
    spectra = spectra[:, 0] + 1j * spectra[:, 1]
    maxIdx = np.argmax(np.abs(spectra), axis=1)
    phase = np.angle(np.take_along_axis(spectra, maxIdx[:, None], axis=1))
    spectra = spectra * np.exp(-1j * phase)
    return np.stack((np.real(spectra), np.imag(spectra)), axis=1)


#***************************************#
#   own nifti eddy current correction   #
#***************************************#
def own_nifti_ecc(data, reference):
    """
        Eddy current correction for MRS data in the NIfTI format. Using the code from suspect.

        @param data -- The MRS data to be corrected.
        @param reference -- The reference data for the correction.

        @returns -- The corrected MRS data.
    """

    if data.shape != reference.shape \
            and reference.ndim > 4:
        raise DimensionsDoNotMatch('Reference and data shape must match'
                                   ' or reference must be single FID.')

    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        if data.shape == reference.shape:
            # reference is the same shape as data, voxel-wise and spectrum-wise iteration
            ref = reference[idx]
        else:
            # only one reference FID, only iterate over spatial voxels.
            ref = reference[idx[0], idx[1], idx[2], :]

        ec_smooth = sliding_gaussian(np.unwrap(np.angle(ref)), 32)
        ecc = np.exp(-1j * ec_smooth)
        corrected_obj[idx] = dd * ecc

    # update processing prov
    processing_info = f'{__name__}.ecc, '
    processing_info += f'reference={reference.filename}.'
    update_processing_prov(corrected_obj, 'Eddy current correction', processing_info)

    return corrected_obj


#********************************#
#   own nifti coil combination   #
#********************************#
def own_nifti_coil_combination(data, reference):
   """
       Simple coil combination for MRS data in the NIfTI format.
       Using amplitude and phase information from the water peak.

       @param data -- The MRS data to be combined.
       @param reference -- The reference data for the combination.

       @returns -- The combined MRS data.
   """
   combined_obj = data.copy(remove_dim='DIM_COIL')
   for ref, idx in reference.iterate_over_dims(dim='DIM_COIL',
                                               iterate_over_space=True,
                                               reduce_dim_index=False):

       # coil-based weighted average and phase correction
       water_amps = np.trapezoid(np.abs(ref), axis=0)
       water_amps = (water_amps / np.sum(water_amps))[np.newaxis, :, np.newaxis]
       water_phases = np.angle(np.trapezoid(ref, axis=0))[np.newaxis, :, np.newaxis]
       data_metab = data[idx] * water_amps / np.exp(1j * water_phases)
       combined_obj[idx] = np.sum(data_metab, axis=-2)   # sum over coils

   # update processing prov
   processing_info = f'{__name__}.coil_combination, '
   processing_info += f'reference={reference.filename}.'
   update_processing_prov(combined_obj, 'Coil combination', processing_info)

   return combined_obj


#*********************************#
#   estimate coil sensitivities   #
#*********************************#
def estimate_csm(data):
    """
        Estimate the coil sensitivity maps (CSM) from the reference data.
        Adapted from FID-A.

        @param data -- The reference data.

        @returns -- The coil sensitivity maps.
    """
    s_raw = data / np.sqrt(np.sum(data * np.conj(data), axis=0))
    Rs = np.einsum('ij,ik->jk', s_raw, np.conj(s_raw))
    csm, _ = eig_power(Rs)
    return csm


#*****************************#
#   eigenvalue power method   #
#*****************************#
def eig_power(R):
    """
        Eigenvalue power method for the coil sensitivity maps (CSM)
        from the autocorrelation matrix.

        @param R -- The reference data.

        @returns -- The coil sensitivity maps.
    """
    rows, cols = R.shape
    N_iterations = 2
    v = np.ones((rows, cols), dtype=complex)
    for _ in range(N_iterations):
        v = np.dot(R, v)
        d = np.sqrt(np.sum(np.abs(v) ** 2, axis=0))
        d[d <= np.finfo(float).eps] = np.finfo(float).eps
        v = v / d
    p1 = np.angle(np.conj(v[:, 0]))
    v = v * np.exp(1j * p1)[:, None]
    return np.conj(v), d


#*********************************#
#   coil combination (adaptive)   #
#*********************************#
def coil_combination_adaptive(data, water=None):
    """
        Coil combination using amplitude and phase information from the water peak.
        Adapted from CIBM.

        @param data -- The MRS data to be combined.
        @param water -- The reference data for the combination (default=None).

        @returns -- The combined MRS data.
    """
    ref = water if water is not None and water.size != 0 else data
    if ref is data:
        print("No water reference provided, using metabolite data as reference")

    # compute the coil sensitivity maps
    ref = np.mean(ref, axis=2)
    phase = np.exp(-1j * np.angle(ref))
    csm = estimate_csm(ref * phase)[:, 0]
    csmsq = np.sum(csm * np.conj(csm), axis=0)
    csm[csm < np.finfo(float).eps] = 1

    def combine(data):
        combined = np.sum(np.conj(csm)[None, :, None] * data * phase[..., None], axis=1) / csmsq
        return combined

    if water is None:
        return combine(data)
    else:
        return combine(data), combine(water)


#****************************************#
#   nifit wrapper for coil combination   #
#****************************************#
def own_nifti_coil_combination_adaptive(data, reference, report=None):
    """
        Nifit wrapper for the adaptive coil combination.

        @param data -- The MRS data to be combined.
        @param reference -- The reference data for the combination.
        @param report -- The report file (default=None).

        @returns -- The combined MRS data.
    """
    combined_data = data.copy(remove_dim='DIM_COIL')
    combined_wat = reference.copy(remove_dim='DIM_COIL')
    for main, idx in data.iterate_over_spatial():
        # coil combination
        data_metab, data_wref = coil_combination_adaptive(main, reference[idx])

        # update data
        combined_data[idx] = data_metab
        combined_wat[idx] = data_wref

    # plot
    if report is not None:
        for main, idx in data.iterate_over_dims(dim='DIM_COIL',
                                                iterate_over_space=True,
                                                reduce_dim_index=False):

            from fsl_mrs.utils.preproc.combine import combine_FIDs_report
            if all([ii == slice(None, None, None) or ii == 0 for ii in idx]):  # first index
                fig = combine_FIDs_report(main,
                                          combined_data[:].mean(-1).squeeze(),
                                          data.bandwidth,
                                          data.spectrometer_frequency[0],
                                          data.nucleus[0],
                                          ncha=data.shape[data.dim_position('DIM_COIL')],
                                          ppmlim=(0.0, 6.0),
                                          method='adaptive',
                                          dim='DIM_COIL',
                                          html=report)

    # update processing prov
    processing_info = f'{__name__}.coil_combination, '
    processing_info += f'reference={reference.filename}.'
    update_processing_prov(combined_data, 'Coil combination', processing_info)

    return combined_data, combined_wat


#*****************************#
#   resample and FIR filter   #
#*****************************#
def resample_signal_fir(data, npoints, ntabs=12):
    """
        Resample the data to npoints using a FIR filter.

        @param data -- The data to be resampled.
        @param npoints -- The number of points to resample to.
        @param ntabs -- The number of filter taps (default=12).

        @returns -- The resampled data.
    """
    from scipy.signal import firwin, lfilter

    # FIR filter design parameters
    downsample_factor = data.shape[1] // npoints
    cutoff_frequency = 1 / (2 * downsample_factor)   # Nyquist

    # FIR filter
    fir_coeffs = firwin(ntabs, cutoff_frequency, window='hamming')
    data = lfilter(fir_coeffs, 1.0, data.squeeze(), axis=1)[:, ::downsample_factor, ...]

    return data


#*********************#
#   resample signal   #
#*********************#
def resample_signal_lp(data, npoints, bandwidth, axis=1):
    """
        Resample the signal to a new sampling frequency.

        @param data -- The signal to be resampled.
        @param npoints -- The number of points to resample to.
        @param bandwidth -- The desired bandwidth.
        @param axis -- The axis along which to resample (default=1).

        @returns -- The resampled signal.
    """
    from scipy.signal import resample, butter, filtfilt

    # low-pass filter
    nyquist = data.shape[axis] / 2
    cutoff = bandwidth / nyquist  # normalized cutoff frequency
    b, a = butter(4, cutoff, btype='low')  # 4th-order Butterworth filter

    # apply the filter and resample along axis 1
    def process_signal(single_signal):
        filtered_signal = filtfilt(b, a, single_signal)
        resampled_signal = resample(filtered_signal, npoints)
        return resampled_signal

    return np.apply_along_axis(process_signal, axis=axis, arr=data.squeeze())