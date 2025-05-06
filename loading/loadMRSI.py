####################################################################################################
#                                            loadMRSI.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/11/23                                                                                #
#                                                                                                  #
# Purpose: Load various formats and data types of MRSI data.                                       #
#                                                                                                  #
####################################################################################################



#*************#
#   imports   #
#*************#
import numpy as np
import os

from spec2nii.Philips.philips import read_sdat, read_spar


#******************************#
#   load MRSI data from SDAT   #
#******************************#
def vax_to_ieee_single_float(data):
    """
    Taken from: VESPA (https://github.com/vespa-mrs/vespa) - BSD license.

    Converts a float in Vax format to IEEE format.

    data should be a single string of chars that have been read in from
    a binary file. These will be processed 4 at a time into float values.
    Thus the total number of byte/chars in the string should be divisible
    by 4.

    Based on VAX data organization in a byte file, we need to do a bunch of
    bitwise operations to separate out the numbers that correspond to the
    sign, the exponent and the fraction portions of this floating point
    number

    role :      S        EEEEEEEE      FFFFFFF      FFFFFFFF      FFFFFFFF
    bits :      1        2      9      10                               32
    bytes :     byte2           byte1               byte4         byte3

    This is taken from the VESPA project source code under a BSD licence.
    """
    f = []
    nfloat = int(len(data) / 4)
    for i in range(nfloat):

        byte2 = data[0 + i * 4]
        byte1 = data[1 + i * 4]
        byte4 = data[2 + i * 4]
        byte3 = data[3 + i * 4]

        # hex 0x80 = binary mask 10000000
        # hex 0x7f = binary mask 01111111

        sign = (byte1 & 0x80) >> 7
        expon = ((byte1 & 0x7f) << 1) + ((byte2 & 0x80) >> 7)
        fract = ((byte2 & 0x7f) << 16) + (byte3 << 8) + byte4

        if sign == 0:
            sign_mult = 1.0
        else:
            sign_mult = -1.0

        if 0 < expon:
            # note 16777216.0 == 2^24
            val = sign_mult * (0.5 + (fract / 16777216.0)) * pow(2.0, expon - 128.0)
            f.append(val)
        elif expon == sign == 0:
            f.append(0)
        else:
            f.append(0)
            # may want to raise an exception here ...

    return f


#******************************#
#   load MRSI data from SDAT   #
#******************************#
def read_sdat_mrsi(filename, dim1, dim2, dim3):
    """
    Read the .sdat file. Altered for MRSI data.

    @param filename -- Path to file.
    @param dim1 -- Number of spectral points.
    @param dim2 -- Number of rows of data.
    @param dim3 -- Number of columns of data.

    @returns -- The data.
    """
    with open(filename, 'rb') as f:
        raw = f.read()

    floats = vax_to_ieee_single_float(raw)
    data_iter = iter(floats)
    complex_iter = (complex(r, i) for r, i in zip(data_iter, data_iter))
    raw_data = np.fromiter(complex_iter, "complex64")

    # raw_data = np.reshape(raw_data, (rows, samples)).T.squeeze()   # change to below!
    raw_data = np.reshape(raw_data, (dim2, dim3, dim1))
    return raw_data


#*********************************#
#   load MRSI data in SPAR/SDAT   #
#*********************************#
def load_MRSI_sdat_spar(path2data):
    """
    Load MRSI dataset in format of Philips spar sdat.

    @param path2data -- The path to the data.

    @returns -- The data (row, col, fid_points).
    """
    params = read_spar(path2data[:-4] + 'spar')
    data = read_sdat_mrsi(path2data[:-4] + 'sdat',
                          params['dim1_pnts'], params['dim2_pnts'], params['dim3_pnts'])
    return data


#*************************#
#   loading LCModel raw   #
#*************************#
def read_LCModel_raw(filename, conjugate=True):
    """
    Read LCModel (.RAW, .raw, and .H2O) file format. Adapted from [1].

    [1] Clarke, W.T., Stagg, C.J., and Jbabdi, S. (2020). FSL-MRS: An end-to-end
        spectroscopy analysis package. Magnetic Resonance in Medicine, 85, 2950 - 2964.

    @param filename -- Path to .RAW/.H2O file.
    @param bool conjugate -- Apply conjugation upon read.

    @returns -- The basis set data/FID and header if possible.
    """
    header = []
    data   = []
    in_header = False
    after_header = False
    with open(filename, 'r') as f:
        for line in f:
            if (line.find('$') > 0):
                in_header = True

            if in_header:
                header.append(line)
            elif after_header:
                data.append(list(map(float, line.split())))

            if line.find('$END') > 0:
                in_header = False
                after_header = True

    # reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j * data[1::2]).astype(complex)

    # LCModel-specific conjugation
    if conjugate:
        data = np.conj(data)

    return data, header


#*******************************************#
#   loading LCModel raw fixed header size   #
#*******************************************#
def read_LCModel_raw_hs(filename, header_size=11, conjugate=True):
    """
    Read LCModel raw format with user-specified header size.

    @param filename -- Path to file.
    @param header_size -- Number of header lines.
    @param bool conjugate -- Apply conjugation upon read.

    @returns -- The basis set data/FID and header if possible.
    """
    header = []
    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= header_size: data.append(list(map(float, line.split())))
            else: header.append(line)

    # reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j * data[1::2]).astype(complex)

    # LCModel-specific conjugation
    if conjugate:
        data = np.conj(data)

    return data, header


#*****************************#
#   load LCModel coraw data   #
#*****************************#
def load_MRSI_LCModel_coraw(path, prefix='sl1_', dim1=44, dim2=44, dim3=512, verbose=False):
    """
    Load data based on LCModel coraw files.

    @param path -- The path to the files.
    @param prefix -- The prefix of the files.
    @param dim1 -- The number of rows of data (default: 44).
    @param dim2 -- The number of columns of data (default: 44).
    @param dim3 -- The number of spectral points (default: 512).
    @param verbose -- Print out information.

    @returns -- An array containing processed spectra data (dim1, dim2, dim3), if files for
                indices are missing then zeros will be return for the missing.
    """
    xinds = range(1, dim1 + 1)  # row size of mrsi dim
    yinds = range(1, dim2 + 1)  # column size of mrsi dim
    data = {'coraw': np.zeros((dim1, dim2, dim3), dtype=np.complex64)}

    for x_ind in xinds:
        for y_ind in yinds:
            # some voxels might be missing...
            try:
                # load raw data
                sname = path + f'/{prefix}{x_ind}-{y_ind}.coraw'
                if os.stat(sname).st_size == 0: continue   # skip empty files
                d, h = read_LCModel_raw_hs(sname, header_size=11)
                data['coraw'][x_ind - 1, y_ind - 1, :] = d

            except FileNotFoundError:
                if verbose >= 1:
                    print(f'No file with name {sname} found. Skipping...')
    return data


#******************************#
#   load LCModel coord data    #
#******************************#
def read_LCModel_coord(path, coord=True, meta=True):
    """
    Load data based on LCModel coord files.

    @param path -- The path to the files.
    @param coord -- Load concentration estimates.
    @param meta -- Load meta data.

    @returns -- The data.
    """
    metabs, concs, crlbs, tcr = [], [], [], []
    fwhm, snr, shift, phase = None, None, None, None

    # go through file and extract all info
    with open(path, 'r') as file:
        concReader = 0
        miscReader = 0

        for line in file:
            if 'lines in following concentration table' in line:
                concReader = int(line.split(' lines')[0])
            elif concReader > 0:  # read concentration table
                concReader -= 1
                values = line.split()

                # check if in header of table
                if values[0] == 'Conc.':
                    continue
                else:
                    try:  # sometimes the fields are fused together with '+'
                        m = values[3]
                        c = float(values[2])
                    except:
                        if 'E+' in values[2]:  # catch scientific notation
                            c = values[2].split('E+')
                            m = str(c[1].split('+')[1:])
                            c = float(c[0] + 'e+' + c[1].split('+')[0])
                        else:
                            if len(values[2].split('+')) > 1:
                                m = str(values[2].split('+')[1:])
                                c = float(values[2].split('+')[0])
                            elif len(values[2].split('-')) > 1:
                                m = str(values[2].split('-')[1:])
                                c = float(values[2].split('-')[0])
                            else:
                                raise ValueError(f'Could not parse {values}')

                    # append to data
                    metabs.append(m)
                    concs.append(float(values[0]))
                    crlbs.append(int(values[1][:-1]))
                    tcr.append(c)
                    continue

            if 'lines in following misc. output table' in line:
                miscReader = int(line.split(' lines')[0])
            elif miscReader > 0:  # read misc. output table
                miscReader -= 1
                values = line.split()

                # extract info
                if 'FWHM' in values:
                    fwhm = float(values[2])
                    snr = float(values[-1])
                elif 'shift' in values:
                    if values[3] == 'ppm':
                        shift = float(values[2][1:])  # negative fuses with '='
                    else:
                        shift = float(values[3])
                elif 'Ph' in values:
                    phase = float(values[1])

    if coord and meta: return metabs, concs, crlbs, tcr, fwhm, snr, shift, phase
    elif coord: return metabs, concs, crlbs, tcr
    elif meta: return fwhm, snr, shift, phase


#******************************#
#   load LCModel coord data    #
#******************************#
def load_MRSI_LCModel_coord(path, prefix='sl1_', dim1=44, dim2=44, verbose=False):
    """
    Load data based on LCModel coord files.

    @param path -- The path to the files.
    @param prefix -- The prefix of the files.
    @param dim1 -- The number of rows of data (default: 44).
    @param dim2 -- The number of columns of data (default: 44).
    @param verbose -- Print out information.

    @returns -- A dict containing LCModel concentration estimates and meta data.
    """
    xinds = range(1, dim1 + 1)  # row size of mrsi dim
    yinds = range(1, dim2 + 1)  # column size of mrsi dim
    data = {
        'coord': [[{'x': x_ind, 'y': y_ind, 'metabs': [], 'concs': [], 'crlbs': [], '/tcr': []}
                   for y_ind in yinds] for x_ind in xinds],
        'meta': [[{'x': x_ind, 'y': y_ind, 'fwhm': None, 'snr': None, 'shift': None, 'phase': None}
                  for y_ind in yinds] for x_ind in xinds]
    }

    for x_ind in xinds:
        for y_ind in yinds:
            # some voxels might be missing...
            try:
                # load raw data
                sname = path + f'/{prefix}{x_ind}-{y_ind}.coord'
                if os.stat(sname).st_size == 0: continue   # skip empty files

                # go through file and extract all info
                m, c, cr, t, f, s, sh, p = read_LCModel_coord(sname, coord=True, meta=True)

                # append to data
                data['coord'][x_ind - 1][y_ind - 1]['metabs'] = m
                data['coord'][x_ind - 1][y_ind - 1]['concs'] = c
                data['coord'][x_ind - 1][y_ind - 1]['crlbs'] = cr
                data['coord'][x_ind - 1][y_ind - 1]['/tcr'] = t

                data['meta'][x_ind - 1][y_ind - 1]['fwhm'] = f
                data['meta'][x_ind - 1][y_ind - 1]['snr'] = s
                data['meta'][x_ind - 1][y_ind - 1]['shift'] = sh
                data['meta'][x_ind - 1][y_ind - 1]['phase'] = p

            except FileNotFoundError:
                if verbose >= 1:
                    print(f'No file with name {sname} found. Skipping...')
    return data


#*****************************#
#   load LCModel MRSI files   #
#*****************************#
def load_MRSI_LCModel(path, params=None, coraw=True, coord=True, verbose=False):
    """
    Load MRSI dataset from the requested formats of LCModel, i.e. coraw and coord.

    @param path -- The path to the data.
    @param params -- The parameters to load the data. If None they will be read from the SPAR file.
                     Otherwise, the dict is required (spec_points, rows, cols):
                     {'dim1_pnts': int, 'dim2_pnts': int, 'dim3_pnts': int}.
    @param coraw -- Load the raw data.
    @param coord -- Load the concentration estimates and meta data.
    @param verbose -- Print out information.

    @returns -- The data.
    """
    if params is None:
        # find spar file in path
        for file in os.listdir(path):
            if file.endswith('.spar'):
                params = read_spar(path + file)
                break

    xinds = range(1, params['dim2_pnts'] + 1)  # row size of mrsi dim
    yinds = range(1, params['dim3_pnts'] + 1)  # column size of mrsi dim
    data = {
        'coraw': np.zeros((params['dim2_pnts'], params['dim3_pnts'], params['dim1_pnts']),
                          dtype=np.complex64),
        'coord': [[{'x': x_ind, 'y': y_ind, 'metabs': [], 'concs': [], 'crlbs': [], '/tcr': []}
                   for y_ind in yinds] for x_ind in xinds],
        'meta': [[{'x': x_ind, 'y': y_ind, 'fwhm': None, 'snr': None, 'shift': None, 'phase': None}
                  for y_ind in yinds] for x_ind in xinds]
    }

    for x_ind in xinds:
        for y_ind in yinds:
            # some voxels might be missing...
            try:
                # load raw data
                if coraw:
                    sname = path + f'/sl1_{x_ind}-{y_ind}.coraw'
                    if os.stat(sname).st_size == 0: continue
                    d, h = read_LCModel_raw_hs(sname, header_size=11)
                    data['coraw'][x_ind - 1, y_ind - 1, :] = d

                # load coord data
                if coord:
                    sname = path + f'/sl1_{x_ind}-{y_ind}.coord'
                    if os.stat(sname).st_size == 0: continue

                    # go through file and extract all info
                    m, c, cr, t, f, s, sh, p = read_LCModel_coord(sname, coord=True, meta=True)

                    # append to data
                    data['coord'][x_ind - 1][y_ind - 1]['metabs'] = m
                    data['coord'][x_ind - 1][y_ind - 1]['concs'] = c
                    data['coord'][x_ind - 1][y_ind - 1]['crlbs'] = cr
                    data['coord'][x_ind - 1][y_ind - 1]['/tcr'] = t

                    data['meta'][x_ind - 1][y_ind - 1]['fwhm'] = f
                    data['meta'][x_ind - 1][y_ind - 1]['snr'] = s
                    data['meta'][x_ind - 1][y_ind - 1]['shift'] = sh
                    data['meta'][x_ind - 1][y_ind - 1]['phase'] = p

            except FileNotFoundError:
                if verbose >= 1:
                    print(f'No file with name {sname} found. Skipping...')
    return data
