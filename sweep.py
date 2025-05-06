####################################################################################################
#                                            sweep.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Sweep parameter definition for Weights & Biases.                                        #
#                                                                                                  #
####################################################################################################

if __name__ == '__main__':

    #*************#
    #   imports   #
    #*************#
    import pytorch_lightning as pl
    import wandb

    # own
    from train import Pipeline


    #**************************#
    #   eliminate randomness   #
    #**************************#
    pl.seed_everything(42)


    #************************************#
    #   configure sweep and parameters   #
    #************************************#
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val_conc_loss',
        'goal': 'minimize',

        'additional_metrics': {
            'name': 'test_loss',
            'goal': 'minimize'
        }
    }

    sweep_parameters = {
        'skip_train': {'values': [False]},
        'online': {'values': [True]},

        # SNF
        'project': {'values': ['SNFforMRS']},
        'model': {'values': ['snf']},  # 'snf', ...
        'loss': {'values': ['rvdb']},  # 'mae', 'mse', ...
        'path2basis': {'values': ['../Data/BasisSets/7T_sLASER_OIT_TE34.basis']},
        'specType': {'values': ['auto']},  # 'auto', 'synth', 'invivo', 'all'
        'ppmlim': {'values': [(0.5, 4.5)]},  # ppm limits for the spectra (used if specType='auto')
        'basisFmt': {'values': ['7tslaser']},  # '', 'cha_philips', 'biggaba', '7tslaser', ...
        'dataType': {'values': ['aumc2']},  # 'aumc', 'aumc2', 'aumc_mod', 'custom', ...
        'arch': {'values': ['mlp']},
        'width': {'values': [512]},
        'depth': {'values': [3]},
        'activation': {'values': ['elu']},  # 'relu', 'elu', 'tanh', ...
        'dropout': {'values': [0.0]},
        'flow_args': {'values': [{
            'decoder': 'model_based',  # 'default', 'model_based'
            'flow': 'SNFVAE',  # 'baseVAE', 'IAFVAE', 'SNFVAE', ...
            'num_flows': 8,
            'beta': 10.0,
        }]},
        # 'max_steps': {'values': [10000]},
        # 'flow_args.loss': {'values': ['rvdb']},
        # 'flow_args.beta': {'values': [1e0, 1e1, 1e2, 1e3, 1e4]},
        # 'flow_args.arch': {'values': ['mlp']},
        # 'flow_args.dropout': {'values': [0.0]},
        # 'flow_args.width': {'values': [512, 1024]},
        # 'flow_args.depth': {'values': [3, 5]},
        # 'flow_args.activation': {'values': ['elu', 'relu']},
        # 'flow_args.num_flows': {'values': [8, 16]},
        # 'flow_args.hidden_dim': {'values': [256, 512]},
        # 'flow_args.num_hidden': {'values': [2, 4]},
        # 'flow_args.num_ortho_vecs': {'values': [8, 16]},
        # 'flow_args.num_householder': {'values': [8, 16]},
    }

    sweep_config['name'] = 'model_sweep'   # sweep name
    sweep_config['parameters'] = sweep_parameters   # add parameters to sweep
    sweep_config['metric']= metric    # add metric to sweep

    # create sweep ID and name project
    wandb.login(key='1cab1ca299a5707053d08b5384403e036bc9eef5')   # add your own key here
    sweep_id = wandb.sweep(sweep_config,
                           project=sweep_parameters['project']['values'][0],  # project name
                           entity='jume')   # your own entity
    # training the model
    pipeline = Pipeline()
    wandb.agent(sweep_id, pipeline.main)
