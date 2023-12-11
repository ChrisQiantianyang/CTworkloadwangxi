# Scripts for ARIMA forecasting model.

from pmdarima import auto_arima
import pandas as pd
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def base_parser():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=str, default='0',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--save_path', type=str,
                        default='../trained_models/ARIMA/')
    parser.add_argument('--exp_name', type=str,
                        default='ARIMA model')
    parser.add_argument('--save_freq', type=int,
                        default=5)
    config = parser.parse_args()
    return config

def load_data(config):
    pass

def run_arima(config):
    pass





if __name__ == '__main__':

    config = base_parser()
    if config.GPU != '-1':
        config.GPU_print = [int(config.GPU.split(',')[0])]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
    else:
        config.GPU = False

    np.random.seed(config.seed)

    config.save_path = os.path.join(config.save_path, config.proxy, config.gold_standard, f'RF_models_forecast_{config.forecasting_horizon}days')
    config.save_path_reports = os.path.join(config.save_path, 'reports')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_reports, exist_ok=True)

    study = optuna.create_study(study_name=config.exp_name,
                                sampler=optuna.samplers.TPESampler(),
                                direction='maximize')


    study.optimize(lambda trial: objective(trial, config), n_jobs=1, callbacks=[max_trial_callback])

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Avg accuracy', trial.value)

    name_csv = os.path.join(config.save_path, 'Best_hyperparameters.csv')

    print('  Params: ')
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f'{config.save_path_reports}/CV_results.csv')
    trials_df.head()

    dic = dict(trial.params)
    dic['value'] = trial.value
    df = pd.DataFrame.from_dict(data=dic, orient='index').to_csv(name_csv, header=False)


