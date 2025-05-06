####################################################################################################
#                                          loadLABRAW.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 20/11/24                                                                                #
#                                                                                                  #
# Purpose: Load Philips LABRAW files. Most of the code is adapted from MATLAB code, specifically   #
#          Brian Welch's loadLABRAW.m and Vicent Boer's loadLABRAW_vb.m.                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os


#**********************#
#   loading SIN file   #
#**********************#
def load_sin_info(filepath, searchstr='nr_measured_channels'):
    """
    Get information from a Philips SIN file. The SIN file contains information about the 
    raw k-space data. The searchstr defines the information to search for in the file.

    @param filepath -- Filepath of the SIN file.
    @param searchstr -- The string to search for in the file.

    @return The information found in the SIN file.
    """
    try:
        with open(filepath, 'r') as sinfid:
            for line in sinfid:
                if searchstr in line:
                    if line[11:31] == searchstr:
                        info = line.split()[-1]
                        break

    except FileNotFoundError:
        print(f"File '{filepath}' not found")
        return None

    return info


#********************#
#   loading LABRAW   #
#********************#
def load_lab_raw(filepath, **kwargs):
    """
    Load a Philips LABRAW file. The LAB file contains hexadecimal labels that describe
    the data in the RAW file. The RAW file contains the raw k-space data. The function
    loads the data into a numpy array and returns the data and the label information.

    Adapted from Brian Welch's loadLABRAW.m and Vicent Boer's loadLABRAW_vb.m.

    @param filepath -- Filepath of the LAB, RAW or filename prefix.
    @param kwargs -- Optional keyword arguments:

    OptionName |  OptionType  |  OptionDescription
    -----------------------------------------------------------------------------------------------
    coil       |  list        |  List of coils to load.
    kx         |  list        |  List of k-space kx samples to load.
    ky         |  list        |  List of k-space ky rows (E1) to load.
    kz         |  list        |  List of k-space kz rows (E2) to load.
    e3         |  list        |  List of k-space 3rd encoding dim to load.
    loc        |  list        |  List of locations to load.
    ec         |  list        |  List of echoes to load.
    dyn        |  list        |  List of dynamics to load.
    ph         |  list        |  List of cardiac phases to load.
    row        |  list        |  List of rows to load.
    mix        |  list        |  List of mixes to load.
    avg        |  list        |  List of averages to load.
    verbose    |  bool        |  If True, print verbose output. Default is False.
    savememory |  bool        |  If True, use single precision instead of double. Default is True.

    @return data -- N-dimensional array holding the raw k-space data.
            info -- Dictionary containing the label information.
    """

    # initialize DATA and INFO to empty arrays/dictionaries
    data = None
    info = {}

    # initialize INFO structure
    info['filename'] = None
    info['loadopts'] = kwargs
    info['dims'] = {}
    info['labels'] = {}
    info['labels_row_index_array'] = None
    info['label_fieldnames'] = None
    info['idx'] = {}
    info['fseek_offsets'] = None
    info['nLabels'] = None
    info['nLoadedLabels'] = None
    info['nDataLabels'] = None
    info['nNormalDataLabels'] = None
    info['datasize'] = None
    info['FRC_NOISE_DATA'] = None

    # parse the filename, it may be the LAB filename, RAW filename or just the filename prefix
    filename = os.path.basename(filepath)
    prefix, ext = os.path.splitext(filename)
    path = os.path.dirname(filepath) + os.sep

    labname = prefix + '.lab'
    rawname = prefix + '.raw'
    info['filename'] = filename

    # open LAB file and read all hexadecimal labels
    with open(path + labname, 'rb') as labfid:
        # read all hexadecimal labels
        unparsed_labels = np.fromfile(labfid, dtype=np.uint32).reshape(-1, 16).T
        info['nLabels'] = unparsed_labels.shape[1]

    # parse hexadecimal labels (inspired by Holger Eggers' readRaw.m.)
    info['labels']['DataSize'] = {'vals': unparsed_labels[0, :]}
    info['labels']['CodedDataSize'] = {'vals': unparsed_labels[1, :]}  # vincent - new in R5
    info['labels']['LeadingDummies'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[1], (2 ** 16 - 1)), 0)}
    info['labels']['TrailingDummies'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[1], (2 ** 32 - 1)), 16)}
    info['labels']['NormalizationFactor'] = {'vals': unparsed_labels[2, :]}  # vincent - new in R5
    info['labels']['SrcCode'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[2], (2 ** 16 - 1)), 0)}
    info['labels']['DstCode'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[2], (2 ** 32 - 1)), 16)}
    info['labels']['SeqNum'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[3], (2 ** 16 - 1)), 0)}
    info['labels']['LabelType'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[3], (2 ** 32 - 1)), 16)}
    info['labels']['ControlType'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[4], (2 ** 8 - 1)), 0)}
    info['labels']['MonitoringFlag'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[4], (2 ** 16 - 1)), 8)}
    info['labels']['MeasurementPhase'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[4], (2 ** 24 - 1)), 16)}
    info['labels']['MeasurementSign'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[4], (2 ** 32 - 1)), 24)}
    info['labels']['GainSetting'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[5], (2 ** 8 - 1)), 0)}
    info['labels']['Spare1'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[5], (2 ** 16 - 1)), 8)}
    info['labels']['Spare2'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[5], (2 ** 32 - 1)), 16)}
    info['labels']['ProgressCnt'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[6], (2 ** 16 - 1)), 0)}
    info['labels']['Mix'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[6], (2 ** 32 - 1)), 16)}
    info['labels']['Dynamic'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[7], (2 ** 16 - 1)), 0)}
    info['labels']['CardiacPhase'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[7], (2 ** 32 - 1)), 16)}
    info['labels']['Echo'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[8], (2 ** 16 - 1)), 0)}
    info['labels']['Location'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[8], (2 ** 32 - 1)), 16)}
    info['labels']['Row'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[9], (2 ** 16 - 1)), 0)}
    info['labels']['ExtraAtrr'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[9], (2 ** 32 - 1)), 16)}
    info['labels']['Measurement'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[10], (2 ** 16 - 1)), 0)}
    info['labels']['E1'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[10], (2 ** 32 - 1)), 16)}
    info['labels']['E2'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[11], (2 ** 16 - 1)), 0)}
    info['labels']['E3'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[11], (2 ** 32 - 1)), 16)}
    info['labels']['RfEcho'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[12], (2 ** 16 - 1)), 0)}
    info['labels']['GradEcho'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[12], (2 ** 32 - 1)), 16)}
    info['labels']['EncTime'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[13], (2 ** 16 - 1)), 0)}
    info['labels']['RandomPhase'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[13], (2 ** 32 - 1)), 16)}
    info['labels']['RRInterval'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[14], (2 ** 16 - 1)), 0)}
    info['labels']['RTopOffset'] = {'vals': np.right_shift(np.bitwise_and(unparsed_labels[14], (2 ** 32 - 1)), 16)}
    info['labels']['ChannelsActive'] = {'vals': unparsed_labels[15]}
    del unparsed_labels

    # find unique values of each label field
    info['label_fieldnames'] = list(info['labels'].keys())
    for k in info['label_fieldnames']:
        info['labels'][k]['uniq'] = np.unique(info['labels'][k]['vals'])

    # calculate fseek offsets
    info['fseek_offsets'] = np.zeros(info['nLabels'], dtype=np.int64)
    info['fseek_offsets'][0] = 512  # add mysterious 512 byte offset to begin reading file

    # for k in range(1, info['nLabels']):
    #     info['fseek_offsets'][k] = (info['fseek_offsets'][k-1] + info['labels']['DataSize']['vals'][k-1] -
    #                                 info['labels']['TrailingDummies']['vals'][k-1] - info['labels']['LeadingDummies']['vals'][k-1])

    # vincent - adapted for R5
    for k in range(1, info['nLabels']):
        info['fseek_offsets'][k] = info['fseek_offsets'][k - 1] + info['labels']['DataSize']['vals'][k - 1]

    info['idx']['no_data'] = np.where(info['labels']['DataSize']['vals'] == 0)[0]
    info['fseek_offsets'][info['idx']['no_data']] = -1

    # find indices of different label control types
    standard_labels = info['labels']['LabelType']['vals'] == 32513
    info['idx']['NORMAL_DATA'] = np.where(info['labels']['ControlType']['vals'] == 0 & standard_labels)[0]
    info['idx']['DC_OFFSET_DATA'] = np.where(info['labels']['ControlType']['vals'] == 1 & standard_labels)[0]
    info['idx']['JUNK_DATA'] = np.where(info['labels']['ControlType']['vals'] == 2 & standard_labels)[0]
    info['idx']['ECHO_PHASE_DATA'] = np.where(info['labels']['ControlType']['vals'] == 3 & standard_labels)[0]
    info['idx']['NO_DATA'] = np.where(info['labels']['ControlType']['vals'] == 4 & standard_labels)[0]
    info['idx']['NEXT_PHASE'] = np.where(info['labels']['ControlType']['vals'] == 5 & standard_labels)[0]
    info['idx']['SUSPEND'] = np.where(info['labels']['ControlType']['vals'] == 6 & standard_labels)[0]
    info['idx']['RESUME'] = np.where(info['labels']['ControlType']['vals'] == 7 & standard_labels)[0]
    info['idx']['TOTAL_END'] = np.where(info['labels']['ControlType']['vals'] == 8 & standard_labels)[0]
    info['idx']['INVALIDATION'] = np.where(info['labels']['ControlType']['vals'] == 9 & standard_labels)[0]
    info['idx']['TYPE_NR_END'] = np.where(info['labels']['ControlType']['vals'] == 10 & standard_labels)[0]
    info['idx']['VALIDATION'] = np.where(info['labels']['ControlType']['vals'] == 11 & standard_labels)[0]
    info['idx']['NO_OPERATION'] = np.where(info['labels']['ControlType']['vals'] == 12 & standard_labels)[0]
    info['idx']['DYN_SCAN_INFO'] = np.where(info['labels']['ControlType']['vals'] == 13 & standard_labels)[0]
    info['idx']['SELECTIVE_END'] = np.where(info['labels']['ControlType']['vals'] == 14 & standard_labels)[0]
    info['idx']['FRC_CH_DATA'] = np.where(info['labels']['ControlType']['vals'] == 15 & standard_labels)[0]
    info['idx']['FRC_NOISE_DATA'] = np.where(info['labels']['ControlType']['vals'] == 16 & standard_labels)[0]
    info['idx']['REFERENCE_DATA'] = np.where(info['labels']['ControlType']['vals'] == 17 & standard_labels)[0]
    info['idx']['DC_FIXED_DATA'] = np.where(info['labels']['ControlType']['vals'] == 18 & standard_labels)[0]
    info['idx']['DNAVIGATOR_DATA'] = np.where(info['labels']['ControlType']['vals'] == 19 & standard_labels)[0]
    info['idx']['FLUSH'] = np.where(info['labels']['ControlType']['vals'] == 20 & standard_labels)[0]
    info['idx']['RECON_END'] = np.where(info['labels']['ControlType']['vals'] == 21 & standard_labels)[0]
    info['idx']['IMAGE_STATUS'] = np.where(info['labels']['ControlType']['vals'] == 22 & standard_labels)[0]
    info['idx']['TRACKING'] = np.where(info['labels']['ControlType']['vals'] == 23 & standard_labels)[0]
    info['idx']['FLUOROSCOPY_TOGGLE'] = np.where(info['labels']['ControlType']['vals'] == 24 & standard_labels)[0]
    info['idx']['REJECTED_DATA'] = np.where(info['labels']['ControlType']['vals'] == 25 & standard_labels)[0]
    info['idx']['UNKNOWN27'] = np.where(info['labels']['ControlType']['vals'] == 27 & standard_labels)[0]
    info['idx']['UNKNOWN28'] = np.where(info['labels']['ControlType']['vals'] == 28 & standard_labels)[0]

    # calculate number of standard, normal data labels
    info['nNormalDataLabels'] = len(info['idx']['NORMAL_DATA'])

    # imension names
    dimnames = ['coil', 'kx', 'ky', 'kz', 'E3', 'loc', 'ec', 'dyn', 'ph', 'row', 'mix', 'avg']
    dimfields = ['N/A', 'N/A', 'E1', 'E2', 'E3', 'Location', 'Echo', 'Dynamic', 'CardiacPhase', 'Row', 'Mix',
                 'Measurement']

    # initialize dimension data to zero
    info['dims']['nCoils'] = 0
    info['dims']['nKx'] = 0
    info['dims']['nKy'] = 0
    info['dims']['nKz'] = 0
    info['dims']['nE3'] = 0
    info['dims']['nLocations'] = 0
    info['dims']['nEchoes'] = 0
    info['dims']['nDynamics'] = 0
    info['dims']['nCardiacPhases'] = 0
    info['dims']['nRows'] = 0
    info['dims']['nMixes'] = 0
    info['dims']['nMeasurements'] = 0

    # # calculate max number of active coils
    # maxChannelsActiveMask = 0
    # for k in info['labels']['ChannelsActive']['uniq']:
    #     maxChannelsActiveMask = maxChannelsActiveMask | k

    # while maxChannelsActiveMask > 0:
    #     print(maxChannelsActiveMask & 1)
    #     if maxChannelsActiveMask & 1:
    #         info['dims']['nCoils'] += 1
    #     maxChannelsActiveMask >>= 1

    # vincent - get number of channels from .sin file
    ncoils = int(load_sin_info(path + prefix + '.sin', searchstr='nr_measured_channels'))
    info['dims']['nCoils'] = ncoils

    # calculate dimensions of normal data
    info['dims']['nKx'] = int(
        max(info['labels']['DataSize']['vals'][info['idx']['NORMAL_DATA']]) / info['dims']['nCoils'] / 2 / 2)
    info['dims']['nKy'] = len(np.unique(info['labels']['E1']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nKz'] = len(np.unique(info['labels']['E2']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nE3'] = len(np.unique(info['labels']['E3']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nLocations'] = len(np.unique(info['labels']['Location']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nEchoes'] = len(np.unique(info['labels']['Echo']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nDynamics'] = len(np.unique(info['labels']['Dynamic']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nCardiacPhases'] = len(np.unique(info['labels']['CardiacPhase']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nRows'] = len(np.unique(info['labels']['Row']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nMixes'] = len(np.unique(info['labels']['Mix']['vals'][info['idx']['NORMAL_DATA']]))
    info['dims']['nMeasurements'] = len(np.unique(info['labels']['Measurement']['vals'][info['idx']['NORMAL_DATA']]))

    # with known possible dimension names, the load options can now be parsed
    for k in dimnames:
        if k in kwargs:
            info['dims'][k] = kwargs[k]

    # return loadopts structure inside info structure
    info['loadopts'] = kwargs

    # find the unique set of values for each dimension name
    info['dims']['coil'] = np.arange(1, info['dims']['nCoils'] + 1)
    info['dims']['kx'] = np.arange(1, info['dims']['nKx'] + 1)
    for k in range(2, len(dimnames)):  # skip coil and kx
        info['dims'][dimnames[k]] = np.unique(info['labels'][dimfields[k]]['vals'][info['idx']['NORMAL_DATA']])

    # find intersection of available dimensions with LOADOPTS dimensions
    for k in dimnames:
        if k in kwargs and kwargs[k] is not None:
            info['dims'][k] = np.intersect1d(kwargs[k], info['dims'][k])

    # calculate data size
    datasize = []
    for k in dimnames:
        datasize.append(len(info['dims'][k]))
    info['datasize'] = datasize

    # throw error if any dimension size is zero
    if 0 in info['datasize']:
        zero_length_str = ' '.join([f"'{dimnames[i]}'" for i, x in enumerate(info['datasize']) if x == 0])
        raise ValueError(f"Size of selected data to load has zero length along dimension(s): {zero_length_str}")

    # skip data loading if only one output argument is provided, return INFO
    if 'data' not in locals():
        info['labels_row_index_array'] = np.arange(1, info['nLabels'] + 1)
        return info

    # create array to hold label row numbers for loaded data, skip the coil and kx dimensions
    info['labels_row_index_array'] = np.zeros(info['datasize'][2:], dtype=np.int64)

    # pre-allocate DATA array
    if kwargs.get('savememory', True):
        data = np.zeros(info['datasize'], dtype=np.complex64)
    else:
        data = np.zeros(info['datasize'], dtype=np.complex128)

    # read RAW data for selected dimension ranges
    with open(path + rawname, 'rb') as fidraw:
        info['nLoadedLabels'] = 0
        raw_data_fread_size = info['dims']['nCoils'] * info['dims']['nKx'] * 2
        rawdata_2d = np.zeros((info['dims']['nCoils'], info['dims']['nKx']), dtype=np.complex128)

        for label_idx in info['idx']['NORMAL_DATA']:
            load_flag = True
            dim_assign_indices_full_array = []

            for k in range(2, len(dimfields)):
                dimval = info['labels'][dimfields[k]]['vals'][label_idx]
                dim_assign_indices = np.where(dimval == info['dims'][dimnames[k]])[0]

                if len(dim_assign_indices) == 0:
                    load_flag = False
                    break
                else:
                    if k > 2:
                        dim_assign_indices_full_array_new = np.zeros(
                            (dim_assign_indices_full_array.shape[0] * len(dim_assign_indices),
                             dim_assign_indices_full_array.shape[1] + 1), dtype=np.int64)
                        mod_base_a = len(dim_assign_indices_full_array)
                        mod_base_b = len(dim_assign_indices)

                        for d in range(len(dim_assign_indices_full_array_new)):
                            dim_assign_indices_full_array_new[d] = np.concatenate(
                                (dim_assign_indices_full_array[d % mod_base_a], [dim_assign_indices[d % mod_base_b]]))
                    else:
                        dim_assign_indices_full_array_new = dim_assign_indices.reshape(-1, 1)
                    dim_assign_indices_full_array = dim_assign_indices_full_array_new

            if load_flag:
                info['nLoadedLabels'] += 1
                byte_offset = info['fseek_offsets'][label_idx]

                fidraw.seek(byte_offset)
                rawdata_1d = np.fromfile(fidraw, dtype=np.int16, count=raw_data_fread_size)

                # phase correction
                RandomPhase = float(info['labels']['RandomPhase']['vals'][label_idx])
                MeasurementPhase = float(info['labels']['MeasurementPhase']['vals'][label_idx])
                c = np.exp(-1j * np.pi * (2 * RandomPhase / (2 ** 16 - 1) + MeasurementPhase / 2))

                # parse into real and imaginary parts for each coil
                rawdata_1d = rawdata_1d.reshape((info['dims']['nCoils'], info['dims']['nKx'], 2))
                rawdata_2d = c * (rawdata_1d[..., 0] + 1j * rawdata_1d[..., 1])

                # account for measurement sign
                if info['labels']['MeasurementSign']['vals'][label_idx]:
                    rawdata_2d = np.fliplr(rawdata_2d)

                # select chosen coils
                rawdata_2d = rawdata_2d[info['dims']['coil'] - 1]

                # select chosen kx
                rawdata_2d = rawdata_2d[:, info['dims']['kx'] - 1]

                # insert rawdata_2d into proper locations of the data array
                for d in range(len(dim_assign_indices_full_array)):
                    dim_assign_str = ','.join([str(x) for x in dim_assign_indices_full_array[d]])
                    # assign index to table_index table
                    exec(f"info['labels_row_index_array'][{dim_assign_str}] = {label_idx}")
                    # assign read image to correct location in data array
                    exec(f"data[:,:,{dim_assign_str}] = rawdata_2d")

        # read FRC noise data
        frc_noise_idx = info['idx']['FRC_NOISE_DATA'][0]
        data_size = info['labels']['DataSize']['vals'][frc_noise_idx]
        n_coils = info['dims']['nCoils']
        frc_noise_samples_per_coil = data_size // (2 * 2 * n_coils)

        info['FRC_NOISE_DATA'] = np.zeros((n_coils, frc_noise_samples_per_coil), dtype=np.complex64)
        byte_offset = info['fseek_offsets'][frc_noise_idx]
        fidraw.seek(byte_offset)

        # read the raw data
        rawdata_1d = np.fromfile(fidraw, dtype=np.int16, count=data_size // 2).astype(np.float64)

        # process the raw data
        for sample in range(frc_noise_samples_per_coil):
            for coil in range(n_coils):
                re_idx = 2 * frc_noise_samples_per_coil * coil + 2 * sample
                im_idx = re_idx + 1
                info['FRC_NOISE_DATA'][coil, sample] = rawdata_1d[re_idx] + 1j * rawdata_1d[im_idx]

    # calculate total raw data blobs
    size_data = data.shape
    max_img_dims = size_data[2:]
    info['nDataLabels'] = np.prod(max_img_dims)

    # if VERBOSE, display execution information
    if kwargs.get('verbose', False):
        print(f"Loaded {info['nLoadedLabels']} of {info['nNormalDataLabels']} available normal data labels")
        tmpstr = ''
        for k in dimnames:
            tmpstr += f", # {k}: {len(info['dims'][k])}"
        print(f"Data contains {info['nDataLabels']} raw labels - {tmpstr[3:]}")

    return data, info


#**********************#
#   FIR downsampling   #
#**********************#
def FIRdownsample(fid, npoints):
    """
    Downsample FID using a FIR filter.
    TODO: Assert the correctness of the implementation.

    @param fid -- The input FID data.
    @param npoints -- The number of points to downsample to.

    @return The downsampled FID data.
    """
    # reshape FID if necessary
    fid = np.squeeze(fid)

    # get dimensions
    ncoils = fid.shape[0]
    noversamples = fid.shape[1]

    # FIR filter coefficients
    C = np.array([-0.0274622659936243, -0.0316256052372452, 0.00600081666037503,
                  0.0910571641138698, 0.194603313644247, 0.266010793957549,
                  0.266010793957549, 0.194603313644247, 0.0910571641138698,
                  0.00600081666037503, -0.0316256052372452, -0.0274622659936243])
    N_tabs = len(C)
    N_delay = N_tabs // 2
    R_fir = 4
    R_hdf = noversamples / npoints / 4

    # undersampling by R_hdf
    reshaped_dims = (ncoils, int(R_hdf), int(noversamples // R_hdf), *fid.shape[2:])
    H_ref_out = np.sum(fid.reshape(reshaped_dims), axis=1)
    if ncoils == 1:
        H_ref_out = H_ref_out.reshape((1, *H_ref_out.shape))

    # FIR filter with coefficients from C
    S_ref_out = np.zeros((ncoils, npoints, *fid.shape[2:]), dtype=np.complex64)
    n = np.arange(npoints)
    n1 = n * R_fir + 1 + N_delay
    nmax = noversamples // R_hdf - 1

    for k in range(N_tabs):
        n2 = n1 - k
        idx = (n2 >= 0) & (n2 < nmax)
        S_ref_out[:, n[idx], ...] += C[k] * H_ref_out[:, n2[idx].astype(int), ...]

    return S_ref_out


#*************#
#   testing   #
#*************#
if __name__ == '__main__':
    filename = '../../Data/DataSets/phantomMRS/rawdata/20220517_114507_SV_PRESS_35.raw'
    data, info = load_lab_raw(filename, verbose=True)

    fid = FIRdownsample(data, 1024)
    print(fid.shape)

    # plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.fft.fft(fid[:, :, 0, :].sum(-1).T, axis=0))
    plt.show()