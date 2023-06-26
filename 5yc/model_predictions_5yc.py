# How to use: 
# command to save the predictions of graz testset of all 9 investigated feature extractors of the 5-year-classification  
# pickle file (test_pkl) should include columns of all 9 features, a column "time_curated" and "status_curated" 
# command: python model_predictions_5yc.py --exp_name=OS_5yc_revision --tissue_type=TUM --save=True --test_pkl=/path/to/pickle_file_from_step6.pkl
# results are saved under /home/ext_julia/pipeline/results/

import os
import re 
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import TtoE_SLP_5yc as TtoE_SLP
from random import choices
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

def predictions(args):
    for fe in ['dino_tcga_features', 'r26_features']:
        args.subgroup = 'all'
        args.feature_column = fe
        test_TtoE_ensemble(args)

def class_label_5(df):
    df['labels'] = 0  # Initialize the "labels" column with default value 0

    # Create conditions for assigning labels
    condition_1 = (df['time_curated'] <= 59) & (df['status_curated'] == 1)
    condition_2 = (df['time_curated'] == 59) & (df['status_curated'] == 0)
    condition_3 = (df['time_curated'] < 59) & (df['status_curated'] == 0)

    # Assign labels based on conditions
    df.loc[condition_1, 'labels'] = 1
    df.loc[condition_2, 'labels'] = 0
    df.loc[condition_3, 'labels'] = 2
    return df

def calculate_ipcw(df):
    time = np.asarray(df.time_curated.tolist())# Time of events or censoring
    event = np.asarray(df.status_curated.tolist())  # Event indicator (1 for event, 0 for censoring)
    label = np.asarray(df.labels.tolist())  
    kmf = KaplanMeierFitter()
    kmf.fit(time, 1-event)
    unique_times = np.unique(time)
    no_censoring_probability = kmf.survival_function_at_times(unique_times).values.flatten() # g 
    dic = dict(zip(unique_times, no_censoring_probability))
    ipcw = []
    for t,e,l in zip(time, event, label):
        if l == 2:
            ipcw.append(0)
        elif l == 1: 
            ipcw.append(1/dic[t])
        else:
            ipcw.append(1/dic[59])
    df['ipcw'] = ipcw
    print(f'IPCW sanity check: {len(df)} ~ {df.ipcw.sum()}')
    return df, dic

def apply_ipcw(df, ipcw_dict, test):
    # if test == 'dachs':
    #     time_column = 'Fumonths'
    #     event_column = 'death_event'
    # else:
    time_column = 'time_curated'
    event_column = 'status_curated'
        
    time = np.asarray(df[time_column].tolist())# Time of events or censoring
    event = np.asarray(df[event_column].tolist())  # Event indicator (1 for event, 0 for censoring)
    label = np.asarray(df.labels.tolist())
    ipcw = []
    for t,e,l in zip(time, event, label):
        if l == 2:
            ipcw.append(0)
        elif l == 1: 
            ipcw.append(1/ipcw_dict[t])
        else:
            ipcw.append(1/ipcw_dict[59])
    df['ipcw'] = ipcw
    return df


def test_TtoE_ensemble(args):
    test_pkl = args.test_pkl  
    df_train = pd.read_pickle(args.train_pkl)
    df_val = pd.read_pickle(args.val_pkl)
    df_test = pd.read_pickle(test_pkl)
    print(df_test[[args.event_column, args.duration_column]])
    
    print(f'Size of Training/Validation/Testset: {len(df_train)}/{len(df_val)}/{len(df_test)}')
    
    # 5 year label: 0 no death within 5 years, 1: death within 5 years, 2: censored  
    df_train = class_label_5(df_train)
    df_val = class_label_5(df_val)
    df_test = class_label_5(df_test)

    df_train, ipcw_dict = calculate_ipcw(df_train)
    df_val = apply_ipcw(df_val, ipcw_dict, args.test)
    df_test = apply_ipcw(df_test, ipcw_dict, args.test)

    df_train = df_train[df_train.labels != 2]
    df_val = df_val[df_val.labels != 2]
    #df_test = df_test[df_test.labels != 2]
    print(f'Exclude censored data (label=2): {len(df_train)}/{len(df_val)}/{len(df_test)}')
    
    df_train[args.duration_column] = df_train[args.duration_column]+1
    df_val[args.duration_column] = df_val[args.duration_column]+1
    #df_test[args.duration_column] = df_test[args.duration_column]
    
    if args.tissue_type == 'TUM':
        exclude_ids = [row.tn_id for i, row in df_test.iterrows() if len(np.where(np.asarray(row.types) == 8)[0]) == 0]
        print('Patients excluded due to tissue type: ', len(exclude_ids))
        df_test = df_test[~df_test[args.patient_column].isin(exclude_ids)]
    elif args.tissue_type == 'STR':
        exclude_ids = [row.tn_id for i, row in df_test.iterrows() if len(np.where(np.asarray(row.types) == 7)[0]) == 0]
        print('Patients excluded due to tissue type: ', len(exclude_ids))
        df_test = df_test[~df_test[args.patient_column].isin(exclude_ids)]
    elif args.tissue_type == 'TUM+STR':
        exclude_ids = [row.tn_id for i, row in df_test.iterrows() if (len(np.where(np.asarray(row.types) == 7)[0])+len(np.where(np.asarray(row.types) == 8)[0])) == 0]
        print('Patients excluded due to tissue type: ', len(exclude_ids))
        df_test = df_test[~df_test[args.patient_column].isin(exclude_ids)]
        
    df_train = pd.concat([df_train, df_val]) 

    # use only the relevant columns to save RAM 
    df_train = df_train[[args.patient_column, args.feature_column, args.event_column, args.duration_column,args.label,'ipcw']]
    df_val = df_val[[args.patient_column, args.feature_column, args.event_column, args.duration_column, args.label,'ipcw' ]]
    df_test = df_test[[args.patient_column, args.feature_column, args.event_column, args.duration_column, args.label,'ipcw' ]]
    

    print('Some sanity checks before training')
    df_train = df_train[~(df_train[args.feature_column].isnull())]
    df_val = df_val[~(df_val[args.feature_column].isnull())]
    print(f'Samples without this feature type are excluded, remaining: {len(df_train)}, {len(df_val)}')

    df_train = df_train[~(df_train[args.event_column].isnull())]
    df_val = df_val[~(df_val[args.event_column].isnull())]
    df_test = df_test[~(df_test[args.event_column].isnull())]

    print(f'Size of Training/Validation/Testset after checks for event and duration: {len(df_train)}/{len(df_val)}/{len(df_test)}')

    df_train[args.duration_column] = df_train[args.duration_column].apply(lambda x:int(x))
    df_val[args.duration_column] = df_val[args.duration_column].apply(lambda x:int(x))
    df_test[args.duration_column] = df_test[args.duration_column].apply(lambda x:int(x))


    # for (integrated) brier score: 
    survival_train = df_train[[args.event_column, args.duration_column]].astype(int)
    survival_test =  df_test[[args.event_column, args.duration_column]].astype(int)
    survival_train[args.event_column] = survival_train[args.event_column].astype(bool)
    survival_test[args.event_column] = survival_test[args.event_column].astype(bool)
    survival_train = survival_train.to_records(index=False)
    survival_test = survival_test.to_records(index=False)   
    
    # 5 checkpoints
    model_ckpts = []
    for fold, version in zip(['0','1', '2', '3', '4'], ['0', '0','0','0', '0']):
        exp_name = args.exp_name + '_' + args.subgroup + '_' + args.tissue_type + '_' + fold +'_' + args.feature_column + '_' + args.attention
        exp_path = args.root+exp_name+'/version_'+version+'/checkpoints/'
        for (dirpath, dirnames, filenames) in os.walk(exp_path):
            file = [f for f in filenames if 'last' not in f]
            model_ckpts.append(exp_path + file[0])
    
    # model 
    m = TtoE_SLP.SlideModel_Ilse(args.num_durations, args.feature_length, args.survnet_l1, args.dropout_survnet, args.survnet_l2, args.lr, args.wd, survival_train, survival_test, args.num_warmup_epochs)

    # dataloader
    ds = TtoE_SLP.SlideDataSet(df_test, args.patient_column, args.feature_column, args.event_column, args.duration_column, 'ipcw', args.nbr_features, args.tissue_type)
    dl = DataLoader(ds, shuffle=False, batch_size=None, batch_sampler=None, num_workers=6)

    # save mean predictions
    preds = []
    for ckpt in model_ckpts: 
        print(ckpt)
        m = m.load_from_checkpoint(ckpt, strict=False)
        m = m.eval() 
        m.freeze()
        m.survival_test = survival_test
        print(f'#parameters of the model {sum(p.numel() for p in m.parameters())}')
        preds_per_model = []
        for counter, (patient, features, event, duration, ipcw) in enumerate(dl): 
            phi, A, duration = m(features, patient, event, duration)
            phi = 1-phi.sigmoid()
            preds_per_model.append(phi)
        preds.append(torch.concat(preds_per_model))
    
    preds = torch.stack(preds)
    ensemble_preds = torch.mean(preds, dim=0)
    #print(ensemble_preds.shape)
    ensemble_preds = ensemble_preds.squeeze(1).numpy()
    if args.save == True:
        df_test['preds'] = ensemble_preds
    df_test.to_pickle('/home/ext_julia/pipeline/results/preds_'+ args.test + '_'+args.feature_column+'_'+args.tissue_type+'_5yc.pkl')

    
    
if __name__ == "__main__":
    #Parser
    parser = argparse.ArgumentParser(description='You can add a description here')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--exp_name', default='OS', type=str)
    parser.add_argument('--subgroup', default='all', type=str)
    parser.add_argument('--test_subgroup', default='all', type=str)
    parser.add_argument('--tissue_type', default='TUM', type=str)
    
    parser.add_argument('--root', default='/home/ext_julia/pipeline/own_models/', type=str)
    
    parser.add_argument('--train_pkl', default='/home/ext_julia/pipeline/training_pkls_dummies/training.pkl', type=str)
    parser.add_argument('--val_pkl', default='/home/ext_julia/pipeline/training_pkls_dummies/validation.pkl', type=str)
    parser.add_argument('--test_pkl', default='/path/to/test_pkl.pkl', type=str)
    parser.add_argument('--test', default='graz', type=str) # internal testset 

    parser.add_argument('--patient_column', default='tn_id', type=str)
    parser.add_argument('--label', default='labels', type=str)
    parser.add_argument('--feature_column', default='rand_features', type=str)
    parser.add_argument('--event_column', default='status_curated', type=str)
    parser.add_argument('--duration_column', default='time_curated', type=str)
    parser.add_argument('--idx_dur_column', default='idx_dur_column', type=str)
    parser.add_argument('--interval_frac_column', default='interval_frac_column', type=str)
    parser.add_argument('--num_durations', default=1, type=int)
    
    parser.add_argument('--feature_length', default=512, type=int)
    parser.add_argument('--survnet_l1', default=256, type=int) # 512
    parser.add_argument('--survnet_l2', default=128, type=int) # 256
    parser.add_argument('--dropout_survnet', default=0.5, type=float)
    parser.add_argument('--reduction_l1', default=256, type=int)
    
    parser.add_argument('--attention', default='ilse', type=str)
    parser.add_argument('--att_depth', default=1, type=int) # parameters for self-attention
    parser.add_argument('--att_heads', default=2, type=int)
    parser.add_argument('--att_dropout', default=0.3, type=float)
    parser.add_argument('--att_ff_dropout', default=0.3, type=float)
    parser.add_argument('--att_dim_head', default=1, type=int)
    
    parser.add_argument('--tissue_check', default=8, type=int)
    parser.add_argument('--tissue_min', default=0, type=int)

    parser.add_argument('--lr', default=1e-5, type=int)
    parser.add_argument('--wd', default=1e-6, type=int)
    parser.add_argument('--nbr_features', default=None, type=int)
    parser.add_argument('--monitor', default="valid/loss", type=str)
    parser.add_argument('--monitor_mode', default="min", type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_warmup_epochs', default=10, type=int) #100
    parser.add_argument('--acc_grad_batches', default=1, type=int)
    parser.add_argument('--gpu', default="0,", type=str)
    parser.add_argument('--save', default=False, type=bool)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    predictions(args)
