####################################################################################################
#                                        simulationDefs.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Simulate a corpus of data sets for a metabolite basis set with a specific range of      #
#          concentration values and noise.                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np


#**************************#
#   concentration ranges   #
#**************************#
normConcs = {
    # taken from de Graaf 2019
    'Ace': {'name': 'Ace', 'low_limit': 0.0, 'up_limit': 0.5},    # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0.1, 'up_limit': 1.6},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0.4, 'up_limit': 1.7},    # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 2.0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 4.5, 'up_limit': 10.5},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 1.0, 'up_limit': 2.0},  # Gamma-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 1.0, 'up_limit': 2.0},    # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 3.0, 'up_limit': 6.0},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 6.0, 'up_limit': 12.5},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0.2, 'up_limit': 1.0},    # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.4, 'up_limit': 1.7},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 1.7, 'up_limit': 3.0},    # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 4.0, 'up_limit': 9.0},    # Myo-Inositol
    'Lac': {'name': 'Lac', 'low_limit': 0.2, 'up_limit': 1.0},    # Lactate
    'NAA': {'name': 'NAA', 'low_limit': 7.5, 'up_limit': 12.0},   # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.5, 'up_limit': 2.5},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0.2, 'up_limit': 1.0},  # Phosphocholine
    'PCr': {'name': 'PCr', 'low_limit': 3.0, 'up_limit': 5.5},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 1.0, 'up_limit': 2.0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 3.0, 'up_limit': 6.0},    # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0.2, 'up_limit': 0.5},  # Scyllo-inositol

    'ATP': {'name': 'ATP', 'low_limit': 2.0, 'up_limit': 4.0},
    'BCAA': {'name': 'BCAA', 'low_limit': 0.0, 'up_limit': 0.5},
    'Tcho': {'name': 'Tcho', 'low_limit': 0.5, 'up_limit': 2.5},
    'EA': {'name': 'EA', 'low_limit': 0.0, 'up_limit': 1.6},
    'Glycogen': {'name': 'Glycogen', 'low_limit': 3.0, 'up_limit': 6.0},
    'Serine': {'name': 'Serine', 'low_limit': 0.2, 'up_limit': 2.0},
    'Thr': {'name': 'Thr', 'low_limit': 0.0, 'up_limit': 0.5},

    # assumed (also depend on implementation)
    'Mac': {'name': 'Mac', 'low_limit': 0.0, 'up_limit': 10.0},   # Macromolecules
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 0.0, 'up_limit': 400.0},  # Macromolecules

    # naming conventions
    'PC': {'name': 'PC', 'low_limit': 0.2, 'up_limit': 1.0},    # Phosphocholine
    'PCh': {'name': 'PCh', 'low_limit': 0.2, 'up_limit': 1.0},
    'Cho': {'name': 'Cho', 'low_limit': 0.2, 'up_limit': 1.0},
    'mI': {'name': 'mI', 'low_limit': 4.0, 'up_limit': 9.0},      # Myo-Inositol
    'mIns': {'name': 'mIns', 'low_limit': 4.0, 'up_limit': 9.0},
    'sI': {'name': 'sI', 'low_limit': 0.2, 'up_limit': 0.5},      # Scyllo-Inositol
    'Scyllo': {'name': 'Scyllo', 'low_limit': 0.2, 'up_limit': 0.5},   #
    'Ser': {'name': 'Ser', 'low_limit': 0.2, 'up_limit': 2.0},    # Serine

    # individual macromolecules
    'MM_09': {'name': 'MM_09', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM_12': {'name': 'MM_12', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM_14': {'name': 'MM_14', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM_20': {'name': 'MM_20', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM_23': {'name': 'MM_23', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM_30': {'name': 'MM_30', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM_34': {'name': 'MM_34', 'low_limit': 0.0, 'up_limit': 100.0},

    'MM092': {'name': 'MM092', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM121': {'name': 'MM121', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM139': {'name': 'MM139', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM167': {'name': 'MM167', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM204': {'name': 'MM204', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM226': {'name': 'MM226', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM270': {'name': 'MM270', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM299': {'name': 'MM299', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM321': {'name': 'MM321', 'low_limit': 0.0, 'up_limit': 100.0},
    'MM375': {'name': 'MM375', 'low_limit': 0.0, 'up_limit': 100.0},
}

# used to mainly for testing
customConcs = {
    'Ace': {'name': 'Ace', 'low_limit': 0., 'up_limit': 0},    # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0., 'up_limit': 0},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0., 'up_limit': 0},    # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 0., 'up_limit': 0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 0, 'up_limit': 0},       # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0, 'up_limit': 0},   # Gamma-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 0., 'up_limit': 0},    # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 0., 'up_limit': 0},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 0., 'up_limit': 0},    # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0., 'up_limit': 0},    # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0., 'up_limit': 0},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0., 'up_limit': 0},    # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 0, 'up_limit': 0},     # Myo-Inositol
    'Lac': {'name': 'Lac', 'low_limit': 0., 'up_limit': 0},    # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 0, 'up_limit': 0},     # Macromolecules
    'NAA': {'name': 'NAA', 'low_limit':14, 'up_limit': 14},     # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0., 'up_limit': 0},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0., 'up_limit': 0},  # Phosphocholine
    'PCr': {'name': 'PCr', 'low_limit': 0., 'up_limit': 0},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0., 'up_limit': 0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 0., 'up_limit': 0},    # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0., 'up_limit': 0},  # Scyllo-inositol

    # naming conventions
    'PCh': {'name': 'PCh', 'low_limit': 0., 'up_limit': 0.},    # Phosphocholine
    'mI': {'name': 'mI', 'low_limit': 0., 'up_limit': 0.},      # Myo-Inositol
    'sI': {'name': 'sI', 'low_limit': 0., 'up_limit': 0.},      # Scyllo-Inositol
    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 0.},   # Scyllo
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 0., 'up_limit': 0.},  # Macromolecules

    # individual macromolecules
    'MM_09': {'name': 'MM_09', 'low_limit': 0.0, 'up_limit': 0.0},
    'MM_12': {'name': 'MM_12', 'low_limit': 0.0, 'up_limit': 0.0},
    'MM_14': {'name': 'MM_14', 'low_limit': 0.0, 'up_limit': 0.0},
    'MM_20': {'name': 'MM_20', 'low_limit': 0.0, 'up_limit': 0.0},
    'MM_23': {'name': 'MM_23', 'low_limit': 0.0, 'up_limit': 0.0},
    'MM_30': {'name': 'MM_30', 'low_limit': 0.0, 'up_limit': 0.0},
    'MM_34': {'name': 'MM_34', 'low_limit': 0.0, 'up_limit': 0.0},
}

# these are obtained by running FSL-MRS MH fitting method on the
# 2016 ISMRM fitting challenge data set
challengeConcs = {
    'Ace': {'name': 'Ace', 'low_limit': 0., 'up_limit': 3.},     # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0., 'up_limit': 8.},     # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0., 'up_limit': 7.},     # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 8.0},   # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 0., 'up_limit': 9.},       # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0., 'up_limit': 9.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 0., 'up_limit': 3.0},    # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 1.0, 'up_limit': 14.0},  # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 4.0, 'up_limit': 15.0},  # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0., 'up_limit': 9.0},    # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0., 'up_limit': 3.0},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0., 'up_limit': 4.0},    # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 1.0, 'up_limit': 12.0},  # Myo-Inositol
    'Lac': {'name': 'Lac', 'low_limit': 0., 'up_limit': 38.0},   # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 0., 'up_limit': 14.0},   # Macromolecules
    'NAA': {'name': 'NAA', 'low_limit':0., 'up_limit': 18.0},    # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0., 'up_limit': 4.0},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0., 'up_limit': 3.0},  # Phosphocholine
    'PCr': {'name': 'PCr', 'low_limit': 0., 'up_limit': 7.0},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0., 'up_limit': 4.0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 0., 'up_limit': 4.0},    # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0., 'up_limit': 2.0},  # Scyllo-inositol

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 2.0},   # Scyllo
    'PCh': {'name': 'PCh', 'low_limit': 0., 'up_limit': 3.0},         # Phosphocholine
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 0., 'up_limit': 400.0},  # Macromolecules

    'mI': {'name': 'mI', 'low_limit': 1., 'up_limit': 12.0},     # myo-Inositol
    'sI': {'name': 'sI', 'low_limit': 0., 'up_limit': 2.0},      # scyllo-Inositol

    # 'Cit': {'name': 'Cit', 'low_limit': 0., 'up_limit': 0.},    # Citrate
    # 'EtOH': {'name': 'EtOH', 'low_limit': 0., 'up_limit': 0.},  # Ethanol
    # 'Phenyl': {'name': 'Phenyl', 'low_limit': 0., 'up_limit': 0.},  # Phenylalanine
    # 'Ser': {'name': 'Ser', 'low_limit': 0., 'up_limit': 0.},    # Serine
    # 'Tyros': {'name': 'Tyros', 'low_limit': 0., 'up_limit': 0.},  # Tyrosine
    # 'bHB': {'name': 'bHB', 'low_limit': 0., 'up_limit': 0.},    # b-Hydroxybutyrate

    # individual macromolecules
    'MM_09': {'name': 'MM_09', 'low_limit': 0.0, 'up_limit': 400.0},
    'MM_12': {'name': 'MM_12', 'low_limit': 0.0, 'up_limit': 400.0},
    'MM_14': {'name': 'MM_14', 'low_limit': 0.0, 'up_limit': 400.0},
    'MM_20': {'name': 'MM_20', 'low_limit': 0.0, 'up_limit': 400.0},
    'MM_23': {'name': 'MM_23', 'low_limit': 0.0, 'up_limit': 400.0},
    'MM_30': {'name': 'MM_30', 'low_limit': 0.0, 'up_limit': 400.0},
    'MM_34': {'name': 'MM_34', 'low_limit': 0.0, 'up_limit': 400.0},
}

aumcConcsLCM = {
    # fitting the AMCU data with LCMoldel
    'Ala': {'name': 'Ala', 'low_limit': 0.0, 'up_limit': 1.0},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0.0, 'up_limit': 3.7},    # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 0.8, 'up_limit': 4.0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 3.9, 'up_limit': 8.0},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0.0, 'up_limit': 4.0},  # Gamma-aminobutyric acid
    'Gln': {'name': 'Gln', 'low_limit': 1.9, 'up_limit': 6.8},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 11.7, 'up_limit': 16.9},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0.0, 'up_limit': 0.5},    # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.0, 'up_limit': 3.2},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0.0, 'up_limit': 1.7},    # Glutathione
    'mIns': {'name': 'mIns', 'low_limit': 5.9, 'up_limit': 10.0},    # Myo-Inositol
    'Lac': {'name': 'Lac', 'low_limit': 0.0, 'up_limit': 2.8},    # Lactate
    'NAA': {'name': 'NAA', 'low_limit': 9.8, 'up_limit': 14.0},   # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.2, 'up_limit': 2.0},  # N-Acetylaspartylglutamic Acid
    'PCh': {'name': 'PCh', 'low_limit': 0.0, 'up_limit': 2.4},  # Phosphocholine
    'PCr': {'name': 'PCr', 'low_limit': 0.9, 'up_limit': 5.2},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0.0, 'up_limit': 5.2},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 1.2, 'up_limit': 3.1},    # Taurine
    'Scyllo': {'name': 'Scyllo', 'low_limit': 0.0, 'up_limit': 0.4},  # Scyllo-inositol
    'Ser': {'name': 'Ser', 'low_limit': 0.0, 'up_limit': 3.6},  # Serine
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 175.7, 'up_limit': 318.9},  # Macromolecules
}

aumcConcsNewton = {
    # fitting the AMCU data with FSL-MRS Newton
    'Ala': {'name': 'Ala', 'low_limit': 0.0, 'up_limit': 0.5},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0.9, 'up_limit': 4.9},    # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 0.0, 'up_limit': 4.8},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 5.0, 'up_limit': 12.3},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0.0, 'up_limit': 2.2},  # Gamma-aminobutyric acid
    'Gln': {'name': 'Gln', 'low_limit': 0.0, 'up_limit': 3.6},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 11.1, 'up_limit': 17.9},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0.0, 'up_limit': 0.2},    # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.2, 'up_limit': 3.6},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 1.0, 'up_limit': 3.6},    # Glutathione
    'mIns': {'name': 'mIns', 'low_limit': 6.5, 'up_limit': 12.1},    # Myo-Inositol
    'Lac': {'name': 'Lac', 'low_limit': 0.1, 'up_limit': 3.1},    # Lactate
    'NAA': {'name': 'NAA', 'low_limit': 9.7, 'up_limit': 16.3},   # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.0, 'up_limit': 2.3},  # N-Acetylaspartylglutamic Acid
    'PCh': {'name': 'PCh', 'low_limit': 0.0, 'up_limit': 2.2},  # Phosphocholine
    'PCr': {'name': 'PCr', 'low_limit': 0.0, 'up_limit': 5.1},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0.0, 'up_limit': 3.5},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 1.5, 'up_limit': 3.4},    # Taurine
    'Scyllo': {'name': 'Scyllo', 'low_limit': 0.0, 'up_limit': 0.6},  # Scyllo-inositol
    'Ser': {'name': 'Ser', 'low_limit': 0.0, 'up_limit': 7.3},  # Serine
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 133.8, 'up_limit': 242.1},  # Macromolecules
}

# AMCU concentration ranges obtained by taking the minimum and maximum values
# from the LCM fitting methods
aumcConcs = {key: {'name': key,
                   'low_limit': min(valLCM['low_limit'], valNewton['low_limit'], val['low_limit']),
                   'up_limit': max(valLCM['up_limit'], valNewton['up_limit'], val['up_limit'])}
             for (key, valLCM), (_, valNewton), (_, val) in
             zip(aumcConcsLCM.items(), aumcConcsNewton.items(), normConcs.items())}

# lower half  (and upper half) of the concentration range AUMC
aumcConcsLH = {key: {'name': key,
                     'low_limit': val['low_limit'],
                     'up_limit': (val['low_limit'] + val['up_limit']) / 2}
               for key, val in aumcConcs.items()}
aumcConcsUH = {key: {'name': key,
                     'low_limit': (val['low_limit'] + val['up_limit']) / 2,
                     'up_limit': val['up_limit']}
               for key, val in aumcConcs.items()}

# mid slice of the concentration range AUMC
aumcConcsMS = {key: {'name': key,
                     'low_limit': (val['low_limit'] + val['up_limit']) / 4,
                     'up_limit': 3 * (val['low_limit'] + val['up_limit']) / 4}
               for key, val in aumcConcs.items()}

# uniform concentration ranges
unifConcs = {key: {'name': key, 'low_limit': 0., 'up_limit': 20}
             for key in normConcs.keys()}

# mean concentration values
meanConcs = {key: {'name': key,
                   'low_limit': (val['low_limit'] + val['up_limit']) / 2,
                   'up_limit': (val['low_limit'] + val['up_limit']) / 2}
             for key, val in normConcs.items()}
meanConcs['Cr'] = {'name': 'Cr', 'low_limit': 6, 'up_limit': 6}
meanConcs['PCr'] = {'name': 'PCr', 'low_limit': 3, 'up_limit': 3}


#***********************#
#   signal parameters   #
#***********************#
perfectParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (2, 2)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0 ,0 , 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 0],  # [mean, std]
}

customParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (2, 2)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0 ,0 , 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 0],  # [mean, std]
}

cleanParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (5, 5)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 5],  # [mean, std]
}

constParams = {
    'dist': 'unif',
    'broadening': [(5, 5), (5, 5)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [5000, 5000],  # [mean, std]
}


normParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (25, 25)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-60, -80, -100, -60, -160, -40], [20, 30, 60, 100, 20, 100]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [0, np.sqrt(2)/2 * 500],  # [low, high]
}

normParamsRWP = {
    'dist': 'unif',
    'broadening': [(2, 2), (15, 15)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.2, 0.2],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-5, 5],  # [low, high]
    'baseline': [[-60, -80, -100, -60, -160, -40], [20, 30, 60, 100, 20, 100]],
    'ownBaseline': None,
    'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]

    # random walk
    'scale': [0, 1000],  # [low, high]
    'smooth': [1, 1000],  # [low, high]
    'limits': [[-10000, 0], [0, 10000]],  # [[low low, low high], [high low, high high]]

    # peaks
    'numPeaks': [0, 5],  # [low, high]
    'peakAmp': [0, 60000],  # [low, high]
    'peakWidth': [0, 100],  # [low, high]
    'peakPhase': [0, 2 * np.pi],  # [low, high]
}

challengeParams = {
    # these are obtained by running FSL-MRS MH fitting method on the
    # 2016 ISMRM fitting challenge data set
    'dist': 'unif',
    'broadening': [(2, 2), (35, 25)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 1.5],  # [low, high]
    'phi1': [-1e-4, 1e-4],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    # 'baseline': [[-3, -4, -5, -3, -8, -2], [1, 2, 3, 5, 1, 5]],
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    'noise': [0, np.sqrt(2)/2 * 800],  # [mean, std] (account for complex Gaussian)
}


mrsiParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (50, 50)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-np.pi / 2, np.pi / 2],  # [low, high]
    'phi1': [-1e-3, 1e-3],  # [low, high]
    'shifting': [-30, 30],  # [low, high]
    'baseline': [[-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [0, np.sqrt(2)/2 * 20],  # [low, high]

    # random walk
    'scale': [0, 20],  # [low, high]
    'smooth': [1, 1000],  # [low, high]
    'limits': [[-100, 0], [0, 100]],  # [[low low, low high], [high low, high high]]

    # peaks
    'numPeaks': [0, 5],  # [low, high]
    'peakAmp': [0, 500],  # [low, high]
    'peakWidth': [0, 50],  # [low, high]
    'peakPhase': [0, 2 * np.pi],  # [low, high]
}


aumcParams = {
    'dist': 'unif',
    'broadening': [(2, 2), (25, 25)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [5, np.sqrt(2) / 2 * 5000],  # [low, high]
}

aumcParams2 = {
    'dist': 'unif',
    'broadening': [(2, 2), (25, 25)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [10, np.sqrt(2) / 2 * 10000],  # [low, high]
}

aumcParamsMod = {
    'dist': 'unif',
    'broadening': [(2, 2), (35, 35)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [10, np.sqrt(2) / 2 * 10000],  # [low, high]
}

aumcParamsMod2 = {
    'dist': 'unif',
    'broadening': [(2, 2), (35, 35)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [20, np.sqrt(2) / 2 * 20000],  # [low, high]
}

aumcParamsModLW = {
    'dist': 'unif',
    'broadening': [(2, 2), (35, 35)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [np.sqrt(2) / 2 * 4000, np.sqrt(2) / 2 * 4000],  # [low, high]
}

aumcParamsRWP = {
    'dist': 'unif',
    'broadening': [(2, 2), (25, 25)],  # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 0.5],  # [low, high]
    'phi1': [-1e-5, 1e-5],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    # 'noise': [0,  np.sqrt(2)/2 * 500],  # [mean, std]
    'noise_mean': [0, 0],  # [low, high]
    'noise_std': [0, np.sqrt(2) / 2 * 50000],  # [low, high]

    # random walk
    'scale': [0, 1000],  # [low, high]
    'smooth': [1, 1000],  # [low, high]
    'limits': [[-10000, 0], [0, 10000]],  # [[low low, low high], [high low, high high]]

    # peaks
    'numPeaks': [0, 5],  # [low, high]
    'peakAmp': [0, 60000],  # [low, high]
    'peakWidth': [0, 100],  # [low, high]
    'peakPhase': [0, 2 * np.pi],  # [low, high]
}