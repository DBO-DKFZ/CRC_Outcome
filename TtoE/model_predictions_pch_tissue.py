# command to save the predictions of graz testset of all 9 investigated feature extractors and 4 tissue variations of the survival prediction (pc hazard)
# command: python model_predictions_pch_tissue.py --exp_name=OS_v2_test3 --save=True --test_pkl=/path/to/pickle_file_from_step6.pkl

import os
import re 
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import argparse
from pycox.models import PCHazard
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pycox.models.utils import pad_col, make_subgrid
import TtoE_SLP
from random import choices
import math 
from lifelines.utils import concordance_index
#metrics
from pycox.evaluation import EvalSurv
from sksurv.metrics import integrated_brier_score

def predict_surv_df(preds, sub, duration_index):
    n = preds.shape[0]
    hazard = F.softplus(preds).view(-1, 1).repeat(1, sub).view(n, -1).div(sub) # Formel 19
    hazard = pad_col(hazard, where='start')
    surv = hazard.cumsum(1).mul(-1).exp() # Formal 20 
    surv = surv.detach().cpu().numpy()
    index = None
    if duration_index is not None:
        index = make_subgrid(duration_index, sub)
    return pd.DataFrame(surv.transpose(), index) # shape [num-duration+1 x samples N]

def predictions(args):
    for fe in ['rand_features', 'image_features','cam_features','sub_features', 'retccl_features', 'ciga_features','dino_features', 'dino_tcga_features', 'r26_features']:
        print(fe)
        for tissue_type in ['STR', 'TUM+STR', 'other', 'ALL']:
            args.subgroup = 'all'
            args.feature_column = fe
            args.tissue_type = tissue_type
            test_TtoE_ensemble(args)
        

def test_TtoE_ensemble(args):
    # graz set: 
    test_pkl = args.test_pkl
        
    df_train = pd.read_pickle(args.train_pkl)
    df_val = pd.read_pickle(args.val_pkl)
    df_test = pd.read_pickle(test_pkl)
    print(df_test[[args.event_column, args.duration_column]])
    
    print(f'Size of Training/Validation/Testset: {len(df_train)}/{len(df_val)}/{len(df_test)}')
    df_train[args.duration_column] = df_train[args.duration_column]+1
    df_val[args.duration_column] = df_val[args.duration_column]+1
    df_test[args.duration_column] = df_test[args.duration_column]
    
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
        
            
    if args.subgroup == 'HR': 
        df_train = df_train[(df_train['stagediag3']==3.0)|(df_train['stagediag3']==4.0)|((df_train['stagediag3']==2.0)&(df_train['T_stage']==4.0))]
        df_val = df_val[(df_val['stagediag3']==3.0)|(df_val['stagediag3']==4.0)|((df_val['stagediag3']==2.0)&(df_val['T_stage']==4.0))]
    elif args.subgroup == "LR":
        df_train = df_train[(df_train['stagediag3']==1.0)|((df_train['stagediag3']==2.0)&((df_train['T_stage']==1.0)|(df_train['T_stage']==2.0)|(df_train['T_stage']==3.0)))]
        df_val = df_val[(df_val['stagediag3']==1.0)|((df_val['stagediag3']==2.0)&((df_val['T_stage']==1.0)|(df_val['T_stage']==2.0)|(df_val['T_stage']==3.0)))]
    else:
        pass
        
    df_train = pd.concat([df_train, df_val]) 

    # use only the relevant columns to save RAM 
    df_train = df_train[[args.patient_column, args.feature_column, args.event_column, args.duration_column]]
    df_val = df_val[[args.patient_column, args.feature_column, args.event_column, args.duration_column]]
    #df_test = df_test[[args.patient_column, args.feature_column, args.event_column, args.duration_column]]
    

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

    print('PCHazard...')
    labtrans = PCHazard.label_transform(args.num_durations)

    get_target = lambda df: (df[args.duration_column].values, df[args.event_column].values)
    y_train_surv = labtrans.fit_transform(*get_target(df_train))
    y_val_surv = labtrans.transform(*get_target(df_val))
    y_test_surv = labtrans.transform(*get_target(df_test))

    df_train[args.idx_dur_column] = y_train_surv[0]
    df_val[args.idx_dur_column] = y_val_surv[0]
    df_test[args.idx_dur_column] = y_test_surv[0]

    df_train[args.interval_frac_column] = y_train_surv[2]
    df_val[args.interval_frac_column] = y_val_surv[2]
    df_test[args.interval_frac_column] = y_test_surv[2]
    
    if args.num_durations !=60:
            print('adjust time_curated for one year')
            devider = int(60/args.num_durations)
            print(devider)
            df_train['time_curated_adj'] = [math.ceil(time/devider) for time in df_train[args.duration_column]]
            df_val['time_curated_adj'] = [math.ceil(time/devider) for time in df_val[args.duration_column]]
            df_test['time_curated_adj'] = [math.ceil(time/devider) for time in df_test[args.duration_column]]
    if args.num_durations !=60:
        time_column = 'time_curated_adj'
    else:
        time_column = 'time_curated'

    # for integrated brier score:
    survival_train = df_train[[args.event_column, time_column]].astype(int)
    survival_test =  df_test[[args.event_column, time_column]].astype(int)
    survival_train[args.event_column] = survival_train[args.event_column].astype(bool)
    survival_test[args.event_column] = survival_test[args.event_column].astype(bool)
    survival_train = survival_train.to_records(index=False)
    survival_test = survival_test.to_records(index=False)   
    
    model_ckpts = []
    for fold, version in zip(['0','1', '2', '3', '4'], ['0', '0','0','0', '0']):
        # args.exp_name: OS_v2_test2, OS_v2_test2_Outputneurons_1,5,10,30
        # args.subgroup: all, HR, LR
        # args.tissue_type: TUM, STR, TUM+STR, other, ALL
        # args.feature_column: rand, image, cam, ciga, retccl, sub, dino
        exp_name = args.exp_name + '_' + args.subgroup + '_' + args.tissue_type + '_' + fold +'_' + args.feature_column + '_' + args.attention
        exp_path = args.root+exp_name+'/version_'+version+'/checkpoints/'
        for (dirpath, dirnames, filenames) in os.walk(exp_path):
            filename = filenames[0]
            model_ckpts.append(exp_path + filename)
    
    if args.attention == 'ilse':
            m = TtoE_SLP.SlideModel_Ilse(labtrans.cuts, args.num_durations, args.feature_length, args.survnet_l1, args.dropout_survnet, args.survnet_l2, args.lr, args.wd, survival_train, survival_test,args.num_warmup_epochs)
    else: 
        print('No module found')
    print(args.nbr_features)
    
    ds = TtoE_SLP.SlideDataSet(df_test, args.patient_column, args.feature_column, args.event_column, args.duration_column, args.idx_dur_column, args.interval_frac_column, args.nbr_features, args.tissue_type)
    #to Do: checken: muss hier nicht auch time_column rein?
    
    
    dl = DataLoader(ds, shuffle=False, batch_size=None, batch_sampler=None, num_workers=6)
    preds = []
    for ckpt in model_ckpts: 
        print(ckpt)
        m = m.load_from_checkpoint(ckpt, strict=False)
        m = m.eval() 
        m.freeze()
        m.survival_test = survival_test
        preds_per_model = []
        for counter, (patient, features, event, duration, idx_duration, interval_frac) in enumerate(dl): 
            phi, A, duration = m(features, patient, event, duration, idx_duration)
            preds_per_model.append(phi)
        preds.append(torch.concat(preds_per_model))
    preds = torch.stack(preds)
    ensemble_preds = torch.mean(preds, dim=0)
    
    surv_df = predict_surv_df(ensemble_preds, sub=1, duration_index=labtrans.cuts)
    ev = EvalSurv(surv_df, np.asarray(df_test[args.duration_column]), np.asarray(df_test[args.event_column]))
    c_index = ev.concordance_td()
    
    if args.save == True:
        surv_df = np.transpose(surv_df.to_numpy())
        surv_list = np.ndarray.tolist(surv_df)
        ensemble_preds_list = ensemble_preds.tolist()
        df_test['preds'] = surv_list
        df_test['logits'] = ensemble_preds_list
        cohort = test_pkl.split('/')[-1].split('_')[0]
        df_test.to_pickle('/home/ext_julia/pipeline/results/preds_'+ args.test + '_'+args.feature_column+'_'+args.tissue_type+'_pch.pkl')
   
    if args.num_durations ==1: 
        durations = np.asarray(df_test[args.duration_column])
        events = np.asarray(df_test[args.event_column])
        ci_last_all = concordance_index(durations, np.asarray(surv_df.iloc[args.num_durations,:]), events)
        print('Confidence intervals')
        c_index_ci = []
        indices = np.arange(0,ensemble_preds.shape[0])
        for i in range(1000):
            draw = choices(indices, k=ensemble_preds.shape[0])
            d = ensemble_preds[draw,:]
            durations = np.asarray(df_test[args.duration_column].astype(int))[draw]
            events =  np.asarray(df_test[args.event_column].astype(bool))[draw]
            surv_df_boot = predict_surv_df(d, sub=1, duration_index=labtrans.cuts)
            #ci_last = concordance_index(durations, d, events)
            ci_last = concordance_index(durations, np.asarray(surv_df_boot.iloc[args.num_durations,:]), events)
            c_index_ci.append(ci_last)
        print('Cindex: ')
        c_index_ci = sorted(c_index_ci)
        print(f'{round(ci_last_all,4)} [{round(c_index_ci[125],4)}, {round(c_index_ci[975],4)}]')
        
    else:
        #surv_df = np.transpose(surv_df.to_numpy())
        n_times = np.arange(m.survival_test[time_column].min(),m.survival_test[time_column].max()+1)
        surv_df = surv_df[:,n_times]
        ibs = integrated_brier_score(survival_train, m.survival_test, surv_df, n_times)
        # confidence intervals
        print('Confidence intervals')
        c_index_ci, ibs_ci = [], []
        indices = np.arange(0,ensemble_preds.shape[0])
        while len(c_index_ci) !=1000:
            draw = choices(indices, k=ensemble_preds.shape[0])
            d = ensemble_preds[draw,:]
            durations = np.asarray(df_test[args.duration_column].astype(int))[draw]
            events =  np.asarray(df_test[args.event_column].astype(bool))[draw]
            surv_df_boot = predict_surv_df(d, sub=1, duration_index=labtrans.cuts)
            ev = EvalSurv(surv_df_boot, durations,events)
            try: 
                c_index_ci.append(ev.concordance_td())
                surv_df_boot = np.transpose(surv_df_boot.to_numpy())
                # ibs
                #print(events.dtype)
                if args.num_durations !=60:
                    devider = int(60/args.num_durations)
                    durations = [math.ceil(time/devider) for time in durations]
                survival_test = np.rec.fromarrays([events, durations], names=["status_curated",time_column])
                n_times = np.arange(survival_test[time_column].min(),survival_test[time_column].max()+1)
                #print(n_times)
                surv_df_boot = surv_df_boot[:,n_times]
                ibs_ci.append(integrated_brier_score(survival_train, survival_test, surv_df_boot, n_times))
            except:
                print('Problems with metrics due to bad drawing of bootstrapping')
                continue

        print('Cindex: ')
        c_index_ci = sorted(c_index_ci)
        print(f'{round(c_index,4)} [{round(c_index_ci[125],4)}, {round(c_index_ci[975],4)}]')
        print('IBS: ') 
        ibs_ci = sorted(ibs_ci)
        print(f'{round(ibs,4)} [{round(ibs_ci[125],4)}, {round(ibs_ci[975],4)}]')

    
    
if __name__ == "__main__":
    #Parser
    parser = argparse.ArgumentParser(description='You can add a description here')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--exp_name', default='OS', type=str)
    parser.add_argument('--subgroup', default='all', type=str)
    parser.add_argument('--test_subgroup', default='all', type=str)
    parser.add_argument('--ensemble_retrained', default=False, type=bool)
    #parser.add_argument('--epoch_list', nargs='+', required=True)
    parser.add_argument('--tissue_type', default='TUM', type=str)
    
    parser.add_argument('--root', default='/home/ext_julia/pipeline/own_models/', type=str)
    
    parser.add_argument('--train_pkl', default='/home/ext_julia/pipeline/training_pkls_dummies/training.pkl', type=str)
    parser.add_argument('--val_pkl', default='/home/ext_julia/pipeline/training_pkls_dummies/validation.pkl', type=str)
    parser.add_argument('--test_pkl', default='/path/to/test_pkl.pkl', type=str)

    #test sets: 
    parser.add_argument('--test', default='graz', type=str) # internal testset 
    
    parser.add_argument('--patient_column', default='tn_id', type=str)
    parser.add_argument('--feature_column', default='rand_features', type=str)
    parser.add_argument('--event_column', default='status_curated', type=str)
    parser.add_argument('--duration_column', default='time_curated', type=str)
    parser.add_argument('--idx_dur_column', default='idx_dur_column', type=str)
    parser.add_argument('--interval_frac_column', default='interval_frac_column', type=str)
    parser.add_argument('--num_durations', default=60, type=int)
    
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