# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import random
import statsmodels.api as sm
from patsy.contrasts import Helmert
from statsmodels.formula.api import ols
import math
from fooof import FOOOF
from scipy.interpolate import make_interp_spline

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne import create_info
from mne.viz import plot_topomap


#%%


subjects = ['Sub-001', 'Sub-002', 'Sub-003', 'Sub-004', 'Sub-005', 'Sub-006', 'Sub-007', 'Sub-008', 'Sub-009', 'Sub-010',
           'Sub-011', 'Sub-012', 'Sub-013', 'Sub-014', 'Sub-015', 'Sub-016', 'Sub-017', 'Sub-018', 'Sub-019', 'Sub-020',
           'Sub-021', 'Sub-022', 'Sub-023', 'Sub-024', 'Sub-025', 'Sub-026', 'Sub-027', 'Sub-028', 'Sub-029', 'Sub-030',
           'Sub-031', 'Sub-032', 'Sub-033', 'Sub-034', 'Sub-035', 'Sub-036', 'Sub-037', 'Sub-038', 'Sub-039', 'Sub-040',
           'Sub-041', 'Sub-042', 'Sub-043', 'Sub-044', 'Sub-045', 'Sub-046', 'Sub-047', 'Sub-048', 'Sub-049', 'Sub-050',
           'Sub-051', 'Sub-052', 'Sub-053', 'Sub-054', 'Sub-055', 'Sub-056', 'Sub-057', 'Sub-058', 'Sub-059', 'Sub-060',
           'Sub-061', 'Sub-062', 'Sub-063', 'Sub-064', 'Sub-065', 'Sub-066', 'Sub-067', 'Sub-068', 'Sub-069', 'Sub-070',
           'Sub-071', 'Sub-072', 'Sub-073', 'Sub-074', 'Sub-075', 'Sub-076', 'Sub-077', 'Sub-078', 'Sub-079', 'Sub-080',
           'Sub-081', 'Sub-082', 'Sub-083', 'Sub-084', 'Sub-085', 'Sub-086', 'Sub-087', 'Sub-088']


#%%

#looping over subjects: 
for sub in subjects:
    
    # conventional power spectral analysis:
    #load preprocessed .set files (after visual inspection and rejection of bad data periods in MATLAB with eeglab)
    data = mne.io.read_raw_eeglab('D:\ds004504\derivatives\%s\eeg\%s_task-eyesclosed_eeg_inspected.set' % (sub, sub))
    total_psd = data.compute_psd(method='welch', fmin = 1, fmax = 45, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    delta_psd = data.compute_psd(method='welch', fmin = 1, fmax = 4, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    theta_psd = data.compute_psd(method='welch', fmin = 4, fmax = 8, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    alpha_psd = data.compute_psd(method='welch', fmin = 8, fmax = 12, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    beta_psd = data.compute_psd(method='welch', fmin = 12, fmax = 30, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    gamma_psd = data.compute_psd(method='welch', fmin = 30, fmax = 45, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    channel_names = total_psd.ch_names
    total_data = total_psd.get_data(picks = 'all')
    delta_data = delta_psd.get_data(picks = 'all')
    theta_data = theta_psd.get_data(picks = 'all')
    alpha_data = alpha_psd.get_data(picks = 'all')
    beta_data = beta_psd.get_data(picks = 'all')
    gamma_data = gamma_psd.get_data(picks = 'all')
    total_cum = np.sum(total_data, axis = 1)
    delta_rel = np.sum(delta_data, axis = 1)/total_cum
    theta_rel = np.sum(theta_data, axis = 1)/total_cum
    alpha_rel = np.sum(alpha_data, axis = 1)/total_cum
    beta_rel = np.sum(beta_data, axis = 1)/total_cum
    gamma_rel = np.sum(gamma_data, axis = 1)/total_cum
    spec_df = pd.DataFrame(channel_names)
    spec_df = spec_df.rename(columns = {0:'channel'})
    spec_df['rel_delta'] = delta_rel
    spec_df['rel_theta'] = theta_rel
    spec_df['rel_alpha'] = alpha_rel
    spec_df['rel_beta'] = beta_rel
    spec_df['rel_gamma'] = gamma_rel
    
    #aperiodic analysis:
    freqs = total_psd.freqs
    tic = -1
    for channel in channel_names:
        tic = tic + 1
        spectrum = total_psd.get_data(picks = channel)
        spect = spectrum.transpose()
        spect = np.concatenate(spect, axis = 0)
        fm = FOOOF(peak_width_limits=[1, 12])
        freq_range = [1, 45]
        fm.fit(freqs, spect, freq_range) 
        
        #extract metrics:
        exp = fm.get_params('aperiodic_params', 'exponent')
        off = fm.get_params('aperiodic_params', 'offset')
        cfs = fm.get_params('peak_params', 'CF')
        pws = fm.get_params('peak_params', 'PW')
        
        cfs = np.array(cfs)
        pws = np.array(pws)

        #Number of fit peaks: 
        n_peaks = fm.n_peaks_

        # Goodness of fit measures
        error = fm.error_
        R_squared = fm.r_squared_
        
        spec_df.loc[tic, 'exponent'] = exp
        spec_df.loc[tic, 'offset'] = off
        spec_df.loc[tic, 'number of fit peaks'] = n_peaks
        spec_df.loc[tic, 'Goodness of fit: error'] = error
        spec_df.loc[tic, 'Goodness of fit: R^2'] = R_squared
        

        try:
            if len(cfs) > 1:
                counter = 0
                for peak in cfs:
                    counter = counter + 1
                    spec_df.loc[tic, 'center frequency %s' % (counter)] = cfs[counter-1]
                    spec_df.loc[tic, 'aperiodic adjusted %s' % (counter)] = pws[counter-1]
        except:
            try:
                spec_df.loc[tic, 'center frequency 1'] = cfs
                spec_df.loc[tic, 'aperiodic adjusted 1'] = pws
            except:
                print('an exception has occured, the cfs variable may be empty')
        
        
        delta_pws = []
        theta_pws = []
        alpha_pws = []
        beta_pws = []
        gamma_pws = []
        
        ap_corr1 = fm.power_spectrum - fm._ap_fit
        fmfreqs = fm.freqs
        ap_corr = pd.DataFrame(fmfreqs)
        ap_corr = ap_corr.rename(columns = {0:'frequency'})
        ap_corr['power'] = ap_corr1
        
        nums = list(range(1, 12))
        for number in nums:
            cf = spec_df['center frequency %s' % (number)].iloc[tic]
            if cf >= 1 and cf <= 4:
                delta_pws.append(spec_df['aperiodic adjusted %s' % (number)].iloc[tic])
            if cf >= 4 and cf <= 8:
                theta_pws.append(spec_df['aperiodic adjusted %s' % (number)].iloc[tic])
            if cf >= 8 and cf <= 12:
                alpha_pws.append(spec_df['aperiodic adjusted %s' % (number)].iloc[tic])
            if cf >= 12 and cf <= 30:
                beta_pws.append(spec_df['aperiodic adjusted %s' % (number)].iloc[tic])
            if cf >= 30 and cf <= 45:
                gamma_pws.append(spec_df['aperiodic adjusted %s' % (number)].iloc[tic])
        
        if len(delta_pws) >= 1:
            spec_df.loc[tic,'adj_delta'] = max(delta_pws)
        else: 
            idx_del = ap_corr.index[ap_corr['frequency'] == min(fmfreqs, key=lambda x:abs(x-2))]
            spec_df.loc[tic,'adj_delta'] = ap_corr.loc[idx_del, 'power'].values[0]
        if len(theta_pws) >= 1:
            spec_df.loc[tic,'adj_theta'] = max(theta_pws)
        else:
            idx_the = ap_corr.index[ap_corr['frequency'] == min(fmfreqs, key=lambda x:abs(x-6))]
            spec_df.loc[tic,'adj_theta'] = ap_corr.loc[idx_the, 'power'].values[0]
        if len(alpha_pws) >= 1:
            spec_df.loc[tic,'adj_alpha'] = max(alpha_pws)
        else:
            idx_alp = ap_corr.index[ap_corr['frequency'] == min(fmfreqs, key=lambda x:abs(x-10))]
            spec_df.loc[tic,'adj_alpha'] = ap_corr.loc[idx_alp, 'power'].values[0]
        if len(beta_pws) >= 1:
            spec_df.loc[tic,'adj_beta'] = max(beta_pws)
        else:
            idx_bet = ap_corr.index[ap_corr['frequency'] == min(fmfreqs, key=lambda x:abs(x-21))]
            spec_df.loc[tic,'adj_beta'] = ap_corr.loc[idx_bet, 'power'].values[0]
        if len(gamma_pws) >= 1:
            spec_df.loc[tic,'adj_gamma'] =  max(gamma_pws)
        else:
            idx_gam = ap_corr.index[ap_corr['frequency'] == min(fmfreqs, key=lambda x:abs(x-37.5))]
            spec_df.loc[tic,'adj_gamma'] = ap_corr.loc[idx_gam, 'power'].values[0]
            
    
    spec_df['subject'] = [sub]*len(channel_names)

    if sub == "Sub-001":
        all_data = spec_df
    else:
        all_data = pd.concat([all_data, spec_df], ignore_index=True, sort=False)


#%%
#assign the correct groups, i.e., for sub-001 to sub-036 AD, sub-037 to sub-65 HC
#sub-066 to sub-088 FTD (see participants.tsv file)
AD_subs = ['Sub-001', 'Sub-002', 'Sub-003', 'Sub-004', 'Sub-005', 'Sub-006', 'Sub-007', 'Sub-008', 'Sub-009', 'Sub-010',
           'Sub-011', 'Sub-012', 'Sub-013', 'Sub-014', 'Sub-015', 'Sub-016', 'Sub-017', 'Sub-018', 'Sub-019', 'Sub-020',
           'Sub-021', 'Sub-022', 'Sub-023', 'Sub-024', 'Sub-025', 'Sub-026', 'Sub-027', 'Sub-028', 'Sub-029', 'Sub-030',
           'Sub-031', 'Sub-032', 'Sub-033', 'Sub-034', 'Sub-035', 'Sub-036']
HC_subs =  ['Sub-037', 'Sub-038', 'Sub-039', 'Sub-040','Sub-041', 'Sub-042', 'Sub-043', 'Sub-044', 
            'Sub-045', 'Sub-046', 'Sub-047', 'Sub-048', 'Sub-049', 'Sub-050','Sub-051', 'Sub-052',
            'Sub-053', 'Sub-054', 'Sub-055', 'Sub-056', 'Sub-057', 'Sub-058', 'Sub-059', 'Sub-060',
            'Sub-061', 'Sub-062', 'Sub-063', 'Sub-064', 'Sub-065']
FTD_subs = ['Sub-066', 'Sub-067', 'Sub-068', 'Sub-069', 'Sub-070','Sub-071', 'Sub-072', 'Sub-073',
            'Sub-074', 'Sub-075', 'Sub-076', 'Sub-077', 'Sub-078', 'Sub-079', 'Sub-080','Sub-081', 
            'Sub-082', 'Sub-083', 'Sub-084', 'Sub-085', 'Sub-086', 'Sub-087', 'Sub-088']


all_data['Group'] = np.nan

for ind in all_data.index:
    if all_data['subject'][ind] in AD_subs:
        all_data['Group'][ind] = 'A'
    elif all_data['subject'][ind] in HC_subs:
        all_data['Group'][ind] = 'C'
    elif all_data['subject'][ind] in FTD_subs:
        all_data['Group'][ind] = 'F'


#unmute the next line to save the dataframe with all the data before the next step, replace "path" with your 
#path where you would like to save the file

#all_data.to_excel('path/all_subs_excel.xlsx', index=False)

#%%
#Linear models in separate R-script (RScript_LinModels)


#%%
#cluster-based permutation test

#I had previously created a separate excel file ('channel_neighbors.xlsx') that has two columns, one in which each row 
#corresponds to one of the 19 channels, and the second column corresponds to the neighbours of that channel, 
#separated with commas. This file needs to be read first. "path" needs to be replaced with your path.
nb_df = pd.read_excel(r"path\channel_neighbors.xlsx")
nb_df = nb_df.sort_values(by = ["channel"])

nb_df = nb_df.reset_index(drop=True)
nb_df.head(5)


#%%
#get dataset and the relevant effect (I am doing the whole permutation test here in an exemplary 
#fashion for the relative alpha power model and the clinical vs control contrast, this has 
#to be repeated for each model and each effect separately)
#"path" needs to be replaced 
dataset = pd.read_excel(r"path\lin_models_results.xlsx")
my_df = dataset[dataset['dep_var'] == "rel_alpha"]
my_df = my_df[my_df['cond'] == "Clin_vs_Ctrl"]
my_df = my_df.sort_values(by = ["electrodes"])
my_df = my_df.reset_index(drop=True)

###Get true cluster sizes:
#prepare dataframe to hold the clusters
clusters = pd.DataFrame(np.nan, index=range(0,19), columns=["chans", "size", "sig"])

rownr = 0
cl_nr = -1

empty_lists = [ [] for _ in range(19) ]
clusters['chans'] = empty_lists

#%%
for chan1 in my_df["electrodes"]:
    idx = my_df.index[my_df['electrodes'] == chan1]
    df_row = my_df.loc[idx]
    df_row = df_row.reset_index(drop=True)
    Ch_exist = clusters["chans"].sum()
    if Ch_exist:
        if isinstance(Ch_exist, list) and chan1 not in Ch_exist and df_row.at[0, "p_val"] < 0.05:
            new_cluster = abs(df_row["t_val"])
            cl_nr = cl_nr + 1
            checked = list([chan1])
            #get the current neighbors: 
            idx = nb_df.index[nb_df['channel'] == chan1]
            nb = nb_df.loc[idx, "neighbors"] 
            x = nb.tolist()
            nb = x[0].split(',')
            for chan2 in nb:
                idx = my_df.index[my_df['electrodes'] == chan2]
                df_row = my_df.loc[idx]
                df_row = df_row.reset_index(drop=True)
                if df_row.at[0, "p_val"] < 0.05 and chan2 not in checked and chan2 not in Ch_exist:
                    new_cluster = new_cluster + abs(df_row["t_val"])
                    checked.append(chan2)
                    idx = nb_df.index[nb_df['channel'] == chan2]
                    nb2 = nb_df.loc[idx, "neighbors"] 
                    x = nb2.tolist()
                    nb2 = x[0].split(',')
                    for chan3 in nb2:
                        idx = my_df.index[my_df['electrodes'] == chan3]
                        df_row = my_df.loc[idx]
                        df_row = df_row.reset_index(drop=True)
                        if df_row.at[0, "p_val"] < 0.05 and chan3 not in checked and chan3 not in Ch_exist:
                            new_cluster = new_cluster + abs(df_row["t_val"])
                            checked.append(chan3)
                            idx = nb_df.index[nb_df['channel'] == chan3]
                            nb3 = nb_df.loc[idx, "neighbors"] 
                            x = nb3.tolist()
                            nb3 = x[0].split(',')
                            for chan4 in nb3:
                                idx = my_df.index[my_df['electrodes'] == chan4]
                                df_row = my_df.loc[idx]
                                df_row = df_row.reset_index(drop=True)
                                if df_row.at[0, "p_val"] < 0.05 and chan4 not in checked and chan4 not in Ch_exist:
                                    new_cluster = new_cluster + abs(df_row["t_val"])
                                    checked.append(chan4)
                                    idx = nb_df.index[nb_df['channel'] == chan4]
                                    nb4 = nb_df.loc[idx, "neighbors"] 
                                    x = nb4.tolist()
                                    nb4 = x[0].split(',')
                                    for chan5 in nb4:
                                        idx = my_df.index[my_df['electrodes'] == chan5]
                                        df_row = my_df.loc[idx]
                                        df_row = df_row.reset_index(drop=True)
                                        if df_row.at[0, "p_val"] < 0.05 and chan5 not in checked and chan5 not in Ch_exist:
                                            new_cluster = new_cluster + abs(df_row["t_val"])
                                            checked.append(chan5)
                                            idx = nb_df.index[nb_df['channel'] == chan5]
                                            nb5 = nb_df.loc[idx, "neighbors"] 
                                            x = nb5.tolist()
                                            nb5 = x[0].split(',')
                                            for chan6 in nb5:
                                                idx = my_df.index[my_df['electrodes'] == chan6]
                                                df_row = my_df.loc[idx]
                                                df_row = df_row.reset_index(drop=True)
                                                if df_row.at[0, "p_val"] < 0.05 and chan6 not in checked and chan6 not in Ch_exist:
                                                    new_cluster = new_cluster + abs(df_row["t_val"])
                                                    checked.append(chan6)
                                                    idx = nb_df.index[nb_df['channel'] == chan6]
                                                    nb6 = nb_df.loc[idx, "neighbors"] 
                                                    x = nb6.tolist()
                                                    nb6 = x[0].split(',')
                                                    for chan7 in nb6:
                                                        idx = my_df.index[my_df['electrodes'] == chan7]
                                                        df_row = my_df.loc[idx]
                                                        df_row = df_row.reset_index(drop=True)
                                                        if df_row.at[0, "p_val"] < 0.05 and chan7 not in checked and chan7 not in Ch_exist:
                                                            new_cluster = new_cluster + abs(df_row["t_val"])
                                                            checked.append(chan6)
    elif df_row.at[0, "p_val"] < 0.05:
        new_cluster = abs(df_row["t_val"])
        cl_nr = cl_nr + 1
        checked = list([chan1])
        #get the current neighbors: 
        idx = nb_df.index[nb_df['channel'] == chan1]
        nb = nb_df.loc[idx, "neighbors"] 
        x = nb.tolist()
        nb = x[0].split(',')
        for chan2 in nb:
            idx = my_df.index[my_df['electrodes'] == chan2]
            df_row = my_df.loc[idx]
            df_row = df_row.reset_index(drop=True)
            if df_row.at[0, "p_val"] < 0.05 and chan2 not in checked and chan2 not in Ch_exist:
                new_cluster = new_cluster + abs(df_row["t_val"])
                checked.append(chan2)
                idx = nb_df.index[nb_df['channel'] == chan2]
                nb2 = nb_df.loc[idx, "neighbors"] 
                x = nb2.tolist()
                nb2 = x[0].split(',')
                for chan3 in nb2:
                    idx = my_df.index[my_df['electrodes'] == chan3]
                    df_row = my_df.loc[idx]
                    df_row = df_row.reset_index(drop=True)
                    if df_row.at[0, "p_val"] < 0.05 and chan3 not in checked and chan3 not in Ch_exist:
                        new_cluster = new_cluster + abs(df_row["t_val"])
                        checked.append(chan3)
                        idx = nb_df.index[nb_df['channel'] == chan3]
                        nb3 = nb_df.loc[idx, "neighbors"] 
                        x = nb3.tolist()
                        nb3 = x[0].split(',')
                        for chan4 in nb3:
                            idx = my_df.index[my_df['electrodes'] == chan4]
                            df_row = my_df.loc[idx]
                            df_row = df_row.reset_index(drop=True)
                            if df_row.at[0, "p_val"] < 0.05 and chan4 not in checked and chan4 not in Ch_exist:
                                new_cluster = new_cluster + abs(df_row["t_val"])
                                checked.append(chan4)
                                idx = nb_df.index[nb_df['channel'] == chan4]
                                nb4 = nb_df.loc[idx, "neighbors"] 
                                x = nb4.tolist()
                                nb4 = x[0].split(',')
                                for chan5 in nb4:
                                    idx = my_df.index[my_df['electrodes'] == chan5]
                                    df_row = my_df.loc[idx]
                                    df_row = df_row.reset_index(drop=True)
                                    if df_row.at[0, "p_val"] < 0.05 and chan5 not in checked and chan5 not in Ch_exist:
                                        new_cluster = new_cluster + abs(df_row["t_val"])
                                        checked.append(chan5)
                                        idx = nb_df.index[nb_df['channel'] == chan5]
                                        nb5 = nb_df.loc[idx, "neighbors"] 
                                        x = nb5.tolist()
                                        nb5 = x[0].split(',')
                                        for chan6 in nb5:
                                            idx = my_df.index[my_df['electrodes'] == chan6]
                                            df_row = my_df.loc[idx]
                                            df_row = df_row.reset_index(drop=True)
                                            if df_row.at[0, "p_val"] < 0.05 and chan6 not in checked and chan6 not in Ch_exist:
                                                new_cluster = new_cluster + abs(df_row["t_val"])
                                                checked.append(chan6)
                                                idx = nb_df.index[nb_df['channel'] == chan6]
                                                nb6 = nb_df.loc[idx, "neighbors"] 
                                                x = nb6.tolist()
                                                nb6 = x[0].split(',')
                                                for chan7 in nb6:
                                                    idx = my_df.index[my_df['electrodes'] == chan7]
                                                    df_row = my_df.loc[idx]
                                                    df_row = df_row.reset_index(drop=True)
                                                    if df_row.at[0, "p_val"] < 0.05 and chan7 not in checked and chan7 not in Ch_exist:
                                                        new_cluster = new_cluster + abs(df_row["t_val"])
                                                        checked.append(chan7)
    #check if new_cluster exists and, if so, add new_cluster to clusters
    if 'new_cluster' in locals():
        clusters.at[cl_nr, 'chans'] = checked
        clusters.loc[cl_nr, 'size'] = new_cluster[0]

##sometimes, as I needed to limit the number of loops, in case of  large clusters, extending over
#most of the scalp, this code gives two separate clusters that are actually belonging together.
#I therefore always checked the clusters manually and summed them if the code separated what should
#have been one cluster 

#%%
###Get Permuted Cluster Sizes, replace "path" with your path
permute_df = pd.read_excel(r"path\all_subs_excel.xlsx")

permute_df = permute_df.sort_values(by = ["subject"])
permute_df = permute_df.reset_index(drop = True)
L = range(0, 19)
elec_df = permute_df.loc[L]
elecs = elec_df["channel"].values.tolist()

permute_df = permute_df.sort_values(by = ["channel"])
permute_df = permute_df.reset_index(drop = True)

permute_df = permute_df.replace({'Group': {'C':3, 'A':2, 'F':1}})

#create list of possible group labels, for that take the first 88 values from the "Group" column in permute_df
one_g_df = permute_df.iloc[:88]
groups = one_g_df['Group']
groups = groups.reset_index(drop=True)
groups = groups.tolist()


levels = [1,2,3]
contrast = Helmert().code_without_intercept(levels) 
cl_size_dist = list()

#%%

for itr in range(1000):
    #randomly draw group labels and reassign them
    new_groups = sorted(groups, key=lambda x: random.random())
    permute_df['Group'] = new_groups*19
    
    #create dataframe to temporarily hold the model results
    model_df = pd.DataFrame(np.nan, index=range(0,19), columns=["channel", "t_CvC", "p_CvC", "t_AvF", "p_AvF"])
    elec_nr = -1
    
    #recompute model results:
    for elec in elecs:
        elec_nr = elec_nr + 1
        #get dataframe containing values of all subjects for the electrode:
        idx = permute_df.index[permute_df['channel'] == elec]
        elec_df = permute_df.loc[idx]
        elec_df = elec_df.reset_index(drop=True)
        
        contrast.matrix[elec_df.Group-1, :][:20]
        
        #calculate model:
        mod = ols("rel_alpha ~ C(Group, Helmert)", data=elec_df) #here, replace the predicted value if required!!! Now for alpha
        res = mod.fit()
        pms = (res.summary2().tables[1])
        
        #get relevant parameters:
        AvFp = pms.iloc[1, 3]
        CvCp = pms.iloc[2, 3]
        AvFt = pms.iloc[1, 2]
        CvCt = pms.iloc[2, 2]
        
        #assign parameters to model_df
        model_df.loc[elec_nr, 'channel'] = elec
        model_df.loc[elec_nr, 'p_AvF'] = AvFp
        model_df.loc[elec_nr, 't_AvF'] = AvFt
        model_df.loc[elec_nr, 'p_CvC'] = CvCp
        model_df.loc[elec_nr, 't_CvC'] = CvCt
        
    #recompute cluster sizes
    perm_cls = pd.DataFrame(np.nan, index=range(0,19), columns=["chans", "size"])
    empty_lists = [ [] for _ in range(19) ]
    perm_cls['chans'] = empty_lists
    
    for chan1 in model_df["channel"]:
        idx = model_df.index[model_df['channel'] == chan1]
        df_row = model_df.loc[idx]
        df_row = df_row.reset_index(drop=True)
        Ch_exist = perm_cls["chans"].sum()
        if Ch_exist:
            if isinstance(Ch_exist, list) and chan1 not in Ch_exist and df_row.at[0, "p_CvC"] < 0.05: #rename "p_CvC" to "p_AvF", when required, here and in all following instances
                new_cluster = abs(df_row["t_CvC"]) #rename "t_CvC" to "t_AvF", when required, here and in all following instances
                cl_nr = cl_nr + 1
                checked = list([chan1])
                #get the current neighbors: 
                idx = nb_df.index[nb_df['channel'] == chan1]
                nb = nb_df.loc[idx, "neighbors"] 
                x = nb.tolist()
                nb = x[0].split(',')
                for chan2 in nb:
                    idx = model_df.index[model_df['channel'] == chan2]
                    df_row = model_df.loc[idx]
                    df_row = df_row.reset_index(drop=True)
                    if df_row.at[0, "p_CvC"] < 0.05 and chan2 not in checked and chan2 not in Ch_exist:
                        new_cluster = new_cluster + abs(df_row["t_CvC"])
                        checked.append(chan2)
                        idx = nb_df.index[nb_df['channel'] == chan2]
                        nb2 = nb_df.loc[idx, "neighbors"] 
                        x = nb2.tolist()
                        nb2 = x[0].split(',')
                        for chan3 in nb2:
                            idx = model_df.index[model_df['channel'] == chan3]
                            df_row = model_df.loc[idx]
                            df_row = df_row.reset_index(drop=True)
                            if df_row.at[0, "p_CvC"] < 0.05 and chan3 not in checked and chan3 not in Ch_exist:
                                new_cluster = new_cluster + abs(df_row["t_CvC"])
                                checked.append(chan3)
                                idx = nb_df.index[nb_df['channel'] == chan3]
                                nb3 = nb_df.loc[idx, "neighbors"] 
                                x = nb3.tolist()
                                nb3 = x[0].split(',')
                                for chan4 in nb3:
                                    idx = model_df.index[model_df['channel'] == chan4]
                                    df_row = model_df.loc[idx]
                                    df_row = df_row.reset_index(drop=True)
                                    if df_row.at[0, "p_CvC"] < 0.05 and chan4 not in checked and chan4 not in Ch_exist:
                                        new_cluster = new_cluster + abs(df_row["t_CvC"])
                                        checked.append(chan4)
                                        idx = nb_df.index[nb_df['channel'] == chan4]
                                        nb4 = nb_df.loc[idx, "neighbors"] 
                                        x = nb4.tolist()
                                        nb4 = x[0].split(',')
                                        for chan5 in nb4:
                                            idx = model_df.index[model_df['channel'] == chan5]
                                            df_row = model_df.loc[idx]
                                            df_row = df_row.reset_index(drop=True)
                                            if df_row.at[0, "p_CvC"] < 0.05 and chan5 not in checked and chan5 not in Ch_exist:
                                                new_cluster = new_cluster + abs(df_row["t_CvC"])
                                                checked.append(chan5)
                                                idx = nb_df.index[nb_df['channel'] == chan5]
                                                nb5 = nb_df.loc[idx, "neighbors"] 
                                                x = nb5.tolist()
                                                nb5 = x[0].split(',')
                                                for chan6 in nb5:
                                                    idx = model_df.index[model_df['channel'] == chan6]
                                                    df_row = model_df.loc[idx]
                                                    df_row = df_row.reset_index(drop=True)
                                                    if df_row.at[0, "p_CvC"] < 0.05 and chan6 not in checked and chan6 not in Ch_exist:
                                                        new_cluster = new_cluster + abs(df_row["t_CvC"])
                                                        checked.append(chan6)
                                                        idx = nb_df.index[nb_df['channel'] == chan6]
                                                        nb6 = nb_df.loc[idx, "neighbors"] 
                                                        x = nb6.tolist()
                                                        nb6 = x[0].split(',')
                                                        for chan7 in nb6:
                                                            idx = model_df.index[model_df['channel'] == chan7]
                                                            df_row = model_df.loc[idx]
                                                            df_row = df_row.reset_index(drop=True)
                                                            if df_row.at[0, "p_CvC"] < 0.05 and chan7 not in checked and chan7 not in Ch_exist:
                                                                new_cluster = new_cluster + abs(df_row["t_CvC"])
                                                                checked.append(chan6)
            #put new_cluster in perm_cls
            perm_cls.at[cl_nr, 'chans'] = checked
            perm_cls.loc[cl_nr, 'size'] = new_cluster[0]
        elif df_row.at[0, "p_CvC"] < 0.05:
            new_cluster = abs(df_row["t_CvC"])
            cl_nr = cl_nr + 1
            checked = list([chan1])
            #get the current neighbors: 
            idx = nb_df.index[nb_df['channel'] == chan1]
            nb = nb_df.loc[idx, "neighbors"] 
            x = nb.tolist()
            nb = x[0].split(',')
            for chan2 in nb:
                idx = model_df.index[model_df['channel'] == chan2]
                df_row = model_df.loc[idx]
                df_row = df_row.reset_index(drop=True)
                if df_row.at[0, "p_CvC"] < 0.05 and chan2 not in checked and chan2 not in Ch_exist:
                    new_cluster = new_cluster + abs(df_row["t_CvC"])
                    checked.append(chan2)
                    idx = nb_df.index[nb_df['channel'] == chan2]
                    nb2 = nb_df.loc[idx, "neighbors"] 
                    x = nb2.tolist()
                    nb2 = x[0].split(',')
                    for chan3 in nb2:
                        idx = model_df.index[model_df['channel'] == chan3]
                        df_row = model_df.loc[idx]
                        df_row = df_row.reset_index(drop=True)
                        if df_row.at[0, "p_CvC"] < 0.05 and chan3 not in checked and chan3 not in Ch_exist:
                            new_cluster = new_cluster + abs(df_row["t_CvC"])
                            checked.append(chan3)
                            idx = nb_df.index[nb_df['channel'] == chan3]
                            nb3 = nb_df.loc[idx, "neighbors"] 
                            x = nb3.tolist()
                            nb3 = x[0].split(',')
                            for chan4 in nb3:
                                idx = model_df.index[model_df['channel'] == chan4]
                                df_row = model_df.loc[idx]
                                df_row = df_row.reset_index(drop=True)
                                if df_row.at[0, "p_CvC"] < 0.05 and chan4 not in checked and chan4 not in Ch_exist:
                                    new_cluster = new_cluster + abs(df_row["t_CvC"])
                                    checked.append(chan4)
                                    idx = nb_df.index[nb_df['channel'] == chan4]
                                    nb4 = nb_df.loc[idx, "neighbors"] 
                                    x = nb4.tolist()
                                    nb4 = x[0].split(',')
                                    for chan5 in nb4:
                                        idx = model_df.index[model_df['channel'] == chan5]
                                        df_row = model_df.loc[idx]
                                        df_row = df_row.reset_index(drop=True)
                                        if df_row.at[0, "p_CvC"] < 0.05 and chan5 not in checked and chan5 not in Ch_exist:
                                            new_cluster = new_cluster + abs(df_row["t_CvC"])
                                            checked.append(chan5)
                                            idx = nb_df.index[nb_df['channel'] == chan5]
                                            nb5 = nb_df.loc[idx, "neighbors"] 
                                            x = nb5.tolist()
                                            nb5 = x[0].split(',')
                                            for chan6 in nb5:
                                                idx = model_df.index[model_df['channel'] == chan6]
                                                df_row = model_df.loc[idx]
                                                df_row = df_row.reset_index(drop=True)
                                                if df_row.at[0, "p_CvC"] < 0.05 and chan6 not in checked and chan6 not in Ch_exist:
                                                    new_cluster = new_cluster + abs(df_row["t_CvC"])
                                                    checked.append(chan6)
                                                    idx = nb_df.index[nb_df['channel'] == chan6]
                                                    nb6 = nb_df.loc[idx, "neighbors"] 
                                                    x = nb6.tolist()
                                                    nb6 = x[0].split(',')
                                                    for chan7 in nb6:
                                                        idx = model_df.index[model_df['channel'] == chan7]
                                                        df_row = model_df.loc[idx]
                                                        df_row = df_row.reset_index(drop=True)
                                                        if df_row.at[0, "p_CvC"] < 0.05 and chan7 not in checked and chan7 not in Ch_exist:
                                                            new_cluster = new_cluster + abs(df_row["t_CvC"])
                                                            checked.append(chan7)
            #add new_cluster to perm_cls
            perm_cls.at[cl_nr, 'chans'] = checked
            perm_cls.loc[cl_nr, 'size'] = new_cluster[0]
    #get the maximum cluster size and add it to cl_size_dist
    max_cl = perm_cls['size'].max()
    cl_size_dist.append(max_cl)
    
    
#%%
#get cutoff to compare original cluster sizes to
cleaned_cluster_dist = [0 if math.isnan(x) else x for x in cl_size_dist]
cutoff = np.percentile(cleaned_cluster_dist, 95)

ind = -1
cl_list = clusters["size"].values.tolist()
cl_list = [x for x in cl_list if str(x) != 'nan']

for i in cl_list:
    ind = ind + 1
    if i > cutoff:
        #clusters.loc[ind, 'sig'] = 's'
        chs = clusters.loc[ind, 'chans']
        print('cluster with channels', chs, 'of size', i, 'is significant')
    else:
        ##clusters.loc[ind, 'sig'] = 'ns'
        chs = clusters.loc[ind, 'chans']
        print('cluster with channels', chs, 'of size', i, 'is not significant')
        

#%%
#Plotting

###Plot F8 Power Spectrum

for sub in subjects:
    
    data = mne.io.read_raw_eeglab('D:\ds004504\derivatives\%s\eeg\%s_task-eyesclosed_eeg_inspected.set' % (sub, sub))
    total_psd = data.compute_psd(method='welch', fmin = 1, fmax = 45, n_fft=1000, n_overlap = 500, n_per_seg = 1000)
    if sub == "Sub-001":
        freqs = total_psd.freqs
        F8_spectra_df = pd.DataFrame(freqs)
    F8spectrum = total_psd.get_data(picks = 'F8')
    F8spect = F8spectrum.transpose()
    F8spect = np.concatenate(F8spect, axis = 0)
    F8_spectra_df['%s' % (sub)] = F8spect
    
#drop the freqs column
F8_spectra_df = F8_spectra_df.drop([0], axis=1)

#split dataframe into the 3 groups
AD_F8_spectra_df = F8_spectra_df.iloc[:, 0:36]
FTD_F8_spectra_df = F8_spectra_df.iloc[:, 65:88]
HC_F8_spectra_df = F8_spectra_df.iloc[:, 36:65]

#get mean across subjects per group df['mean'] = df.mean(axis=1)
AD_F8_mean = np.log(AD_F8_spectra_df.mean(axis=1))
FTD_F8_mean = np.log(FTD_F8_spectra_df.mean(axis=1))
HC_F8_mean = np.log(HC_F8_spectra_df.mean(axis=1))


# Dataset
x = np.log(freqs[4:80])

y1 = AD_F8_mean[4:80]
y2 = FTD_F8_mean[4:80]
y3 = HC_F8_mean[4:80]

#for y1:
X_Y_Spline = make_interp_spline(x, y1)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, color = '#ff7d33')

#for y2:
X_Y_Spline = make_interp_spline(x, y2)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, color = '#6cdfae')

#for y3:
X_Y_Spline = make_interp_spline(x, y3)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, color = '#79aff4')

#in order to show the theta and alpha band in different shades, define borders:
x1 = np.log(4)
x2 = np.log(8)
x3 = np.log(12)
#and show them:
plt.axvspan(x2, x3, alpha=0.2, color='#808080')
plt.axvspan(x1, x2, alpha=0.5, color='#EADDCA')

plt.legend(['AD', 'FTD', 'HC'], fontsize=16)
plt.title("Average Power Spectra F8", fontsize=18)
plt.xlabel("Log(Frequency in Hz)", fontsize=18)
plt.ylabel("Log(Power in V*e⁻¹¹)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


#%%
##Plot F8 offset values as boxplot
#first make dataframe "data_df", which has the offset values of F8 for each group in one column 
#replace "path" with your path
all_subs = pd.read_excel(r"path\all_subs_excel.xlsx")

F8_df1 = all_subs[all_subs['channel'] == 'F8']
F8_HC = F8_df1[F8_df1['Group']== 'C']
F8_FTD = F8_df1[F8_df1['Group']== 'F']
F8_AD = F8_df1[F8_df1['Group']== 'A']

#F8_df = data_df[["F8_offset_AD", "F8_offset_FTD", "F8_offset_HC"]]
#F8_data = [F8_df["F8_offset_AD"].dropna(), F8_df["F8_offset_FTD"].dropna(), F8_df["F8_offset_HC"].dropna()]
F8_data = [F8_HC["offset"].dropna(), F8_FTD["offset"].dropna(), F8_AD["offset"].dropna()]
colors = ['#ff7d33', '#6cdfae', '#79aff4']

with plt.style.context("seaborn-white"):
    fig = plt.figure(figsize =(3, 3),  edgecolor='b')
    _ax = fig.add_axes([0, 0, 1, 1]) 
    
    bplot = plt.boxplot(F8_data, patch_artist=True)
    
    plt.xlabel("")
    plt.ylabel("Offset", fontsize=18)
    plt.xticks([1, 2, 3],['AD', 'FTD', 'HC'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis = 'y')
    
    for median in bplot['medians']:
        median.set_color('black')
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.show()


#%%Topoplots
#before doing the topoplots, I created a separate column in the lin_models_results.xlsx file
#calles "sig", which has an "s" for every significant channel, and an "ns" for every non-sig.
#channel, based on the permutation test results

#The code is again exemplary for the clinical vs control contrast for the relative alpha power
#model

#load dataset
#replace "path" with your path
data = pd.read_excel(r"path\lin_models_results.xlsx")
my_df = data.sort_values(by = ["electrodes"])
my_df = my_df.reset_index(drop=True)
my_df = my_df[my_df['dep_var']== 'rel_alpha']
my_df = my_df[my_df['cond']== 'Clin_vs_Ctrl']


sig_mask1 = my_df[["sig"]].to_numpy()
sig_mask = [
    x
    for xs in sig_mask1
    for x in xs
]
sig_mask = [1 if i == 's' else 0 for i in sig_mask]
sig_mask = np.asarray(sig_mask)

dat_arr1 = my_df[["beta_val"]].values
dat_arr = [
    x
    for xs in dat_arr1
    for x in xs
]

placeholder = np.random.random((25, 25))


ch_arr = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
info = create_info(ch_names=ch_arr, sfreq=512, ch_types='eeg')
info.set_montage('standard_1005')


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10),
                         gridspec_kw=dict(height_ratios=[4, 3]))

axes[0].imshow(placeholder)
axes[0].axis('off')

im,_ = plot_topomap(dat_arr, info, mask = sig_mask, axes=axes[1], show=False) 

plt.title("Relative Alpha", fontsize=20)

divider = make_axes_locatable(axes[1])
cax = divider.append_axes('right', size='5%', pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=16) 

plt.show()



#%%
#Machine learning part:
    
#import libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd  # used to easily load CSV files (and many other file formats)

from sklearn.model_selection import train_test_split  # used to automatically split the dataset
from sklearn.preprocessing import StandardScaler  # used to automatically standardize the input data

# classifier modules from scikit learn:
from sklearn.ensemble import RandomForestClassifier

# grid-search and cross-validation modules:
from sklearn.model_selection import cross_val_score

# metrics modules from scikit learn:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import RocCurveDisplay
#from sklearn.preprocessing import label_binarize  # used to produce ROC curves in multi-class settings
from sklearn.preprocessing import LabelBinarizer


#for oversampling:
from imblearn.over_sampling import SMOTE

#for optimization:
import optuna


#%%
#load data again, replace "path" with your path
df1 = pd.read_excel(r"path\all_subs_excel.xlsx")

#make 2 input dataframes for the two classifiers
df_X1 = df1[['subject', 'Group', 'channel', 'rel_delta', 'rel_theta', 'rel_alpha', 'rel_beta', 'rel_gamma']]
df_X2 = df1[['subject', 'Group', 'channel', 'exponent', 'offset', 'adj_delta', 'adj_theta', 'adj_alpha', 'adj_beta', 'adj_gamma']] 

#make new columns for each electrode
Fp1_df1 = df_X1.loc[df_X1['channel'] == 'Fp1']
Fp2_df1 = df_X1.loc[df_X1['channel'] == 'Fp2']
F3_df1 = df_X1.loc[df_X1['channel'] == 'F3']
F4_df1 = df_X1.loc[df_X1['channel'] == 'F4']
C3_df1 = df_X1.loc[df_X1['channel'] == 'C3']
C4_df1 = df_X1.loc[df_X1['channel'] == 'C4']
P3_df1 = df_X1.loc[df_X1['channel'] == 'P3']
P4_df1 = df_X1.loc[df_X1['channel'] == 'P4']
O1_df1 = df_X1.loc[df_X1['channel'] == 'O1']
O2_df1 = df_X1.loc[df_X1['channel'] == 'O2']
F7_df1 = df_X1.loc[df_X1['channel'] == 'F7']
F8_df1 = df_X1.loc[df_X1['channel'] == 'F8']
T3_df1 = df_X1.loc[df_X1['channel'] == 'T3']
T4_df1 = df_X1.loc[df_X1['channel'] == 'T4']
T5_df1 = df_X1.loc[df_X1['channel'] == 'T5']
T6_df1 = df_X1.loc[df_X1['channel'] == 'T6']
Fz_df1 = df_X1.loc[df_X1['channel'] == 'Fz']
Cz_df1 = df_X1.loc[df_X1['channel'] == 'Cz']
Pz_df1 = df_X1.loc[df_X1['channel'] == 'Pz']

Fp1_df2 = df_X2.loc[df_X2['channel'] == 'Fp1']
Fp2_df2 = df_X2.loc[df_X2['channel'] == 'Fp2']
F3_df2 = df_X2.loc[df_X2['channel'] == 'F3']
F4_df2 = df_X2.loc[df_X2['channel'] == 'F4']
C3_df2 = df_X2.loc[df_X2['channel'] == 'C3']
C4_df2 = df_X2.loc[df_X2['channel'] == 'C4']
P3_df2 = df_X2.loc[df_X2['channel'] == 'P3']
P4_df2 = df_X2.loc[df_X2['channel'] == 'P4']
O1_df2 = df_X2.loc[df_X2['channel'] == 'O1']
O2_df2 = df_X2.loc[df_X2['channel'] == 'O2']
F7_df2 = df_X2.loc[df_X2['channel'] == 'F7']
F8_df2 = df_X2.loc[df_X2['channel'] == 'F8']
T3_df2 = df_X2.loc[df_X2['channel'] == 'T3']
T4_df2 = df_X2.loc[df_X2['channel'] == 'T4']
T5_df2 = df_X2.loc[df_X2['channel'] == 'T5']
T6_df2 = df_X2.loc[df_X2['channel'] == 'T6']
Fz_df2 = df_X2.loc[df_X2['channel'] == 'Fz']
Cz_df2 = df_X2.loc[df_X2['channel'] == 'Cz']
Pz_df2 = df_X2.loc[df_X2['channel'] == 'Pz']


Fp1_df1 = Fp1_df1.rename(columns={"rel_delta": "rel_delta_Fp1", "rel_theta": "rel_theta_Fp1", "rel_alpha": "rel_alpha_Fp1", "rel_beta": "rel_beta_Fp1", "rel_gamma": "rel_gamma_Fp1"})
Fp1_df2 = Fp1_df2.rename(columns={"exponent": "exponent_Fp1", "offset": "offset_Fp1", "adj_delta": "adj_delta_Fp1", "adj_theta": "adj_theta_Fp1", "adj_alpha": "adj_alpha_Fp1", "adj_beta": "adj_beta_Fp1", "adj_gamma": "adj_gamma_Fp1"})

Fp2_df1 = Fp2_df1.rename(columns={"rel_delta": "rel_delta_Fp2", "rel_theta": "rel_theta_Fp2", "rel_alpha": "rel_alpha_Fp2", "rel_beta": "rel_beta_Fp2", "rel_gamma": "rel_gamma_Fp2"})
Fp2_df2 = Fp2_df2.rename(columns={"exponent": "exponent_Fp2", "offset": "offset_Fp2", "adj_delta": "adj_delta_Fp2", "adj_theta": "adj_theta_Fp2", "adj_alpha": "adj_alpha_Fp2", "adj_beta": "adj_beta_Fp2", "adj_gamma": "adj_gamma_Fp2"})
F3_df1 = F3_df1.rename(columns={"rel_delta": "rel_delta_F3", "rel_theta": "rel_theta_F3", "rel_alpha": "rel_alpha_F3", "rel_beta": "rel_beta_F3", "rel_gamma": "rel_gamma_F3"})
F3_df2 = F3_df2.rename(columns={"exponent": "exponent_F3", "offset": "offset_F3", "adj_delta": "adj_delta_F3", "adj_theta": "adj_theta_F3", "adj_alpha": "adj_alpha_F3", "adj_beta": "adj_beta_F3", "adj_gamma": "adj_gamma_F3"})
F4_df1 = F4_df1.rename(columns={"rel_delta": "rel_delta_F4", "rel_theta": "rel_theta_F4", "rel_alpha": "rel_alpha_F4", "rel_beta": "rel_beta_F4", "rel_gamma": "rel_gamma_F4"})
F4_df2 = F4_df2.rename(columns={"exponent": "exponent_F4", "offset": "offset_F4", "adj_delta": "adj_delta_F4", "adj_theta": "adj_theta_F4", "adj_alpha": "adj_alpha_F4", "adj_beta": "adj_beta_F4", "adj_gamma": "adj_gamma_F4"})
C3_df1 = C3_df1.rename(columns={"rel_delta": "rel_delta_C3", "rel_theta": "rel_theta_C3", "rel_alpha": "rel_alpha_C3", "rel_beta": "rel_beta_C3", "rel_gamma": "rel_gamma_C3"})
C3_df2 = C3_df2.rename(columns={"exponent": "exponent_C3", "offset": "offset_C3", "adj_delta": "adj_delta_C3", "adj_theta": "adj_theta_C3", "adj_alpha": "adj_alpha_C3", "adj_beta": "adj_beta_C3", "adj_gamma": "adj_gamma_C3"})
C4_df1 = C4_df1.rename(columns={"rel_delta": "rel_delta_C4", "rel_theta": "rel_theta_C4", "rel_alpha": "rel_alpha_C4", "rel_beta": "rel_beta_C4", "rel_gamma": "rel_gamma_C4"})
C4_df2 = C4_df2.rename(columns={"exponent": "exponent_C4", "offset": "offset_C4", "adj_delta": "adj_delta_C4", "adj_theta": "adj_theta_C4", "adj_alpha": "adj_alpha_C4", "adj_beta": "adj_beta_C4", "adj_gamma": "adj_gamma_C4"})
P3_df1 = P3_df1.rename(columns={"rel_delta": "rel_delta_P3", "rel_theta": "rel_theta_P3", "rel_alpha": "rel_alpha_P3", "rel_beta": "rel_beta_P3", "rel_gamma": "rel_gamma_P3"})
P3_df2 = P3_df2.rename(columns={"exponent": "exponent_P3", "offset": "offset_P3", "adj_delta": "adj_delta_P3", "adj_theta": "adj_theta_P3", "adj_alpha": "adj_alpha_P3", "adj_beta": "adj_beta_P3", "adj_gamma": "adj_gamma_P3"})
P4_df1 = P4_df1.rename(columns={"rel_delta": "rel_delta_P4", "rel_theta": "rel_theta_P4", "rel_alpha": "rel_alpha_P4", "rel_beta": "rel_beta_P4", "rel_gamma": "rel_gamma_P4"})
P4_df2 = P4_df2.rename(columns={"exponent": "exponent_P4", "offset": "offset_P4", "adj_delta": "adj_delta_P4", "adj_theta": "adj_theta_P4", "adj_alpha": "adj_alpha_P4", "adj_beta": "adj_beta_P4", "adj_gamma": "adj_gamma_P4"})
O1_df1 = O1_df1.rename(columns={"rel_delta": "rel_delta_O1", "rel_theta": "rel_theta_O1", "rel_alpha": "rel_alpha_O1", "rel_beta": "rel_beta_O1", "rel_gamma": "rel_gamma_O1"})
O1_df2 = O1_df2.rename(columns={"exponent": "exponent_O1", "offset": "offset_O1", "adj_delta": "adj_delta_O1", "adj_theta": "adj_theta_O1", "adj_alpha": "adj_alpha_O1", "adj_beta": "adj_beta_O1", "adj_gamma": "adj_gamma_O1"})
O2_df1 = O2_df1.rename(columns={"rel_delta": "rel_delta_O2", "rel_theta": "rel_theta_O2", "rel_alpha": "rel_alpha_O2", "rel_beta": "rel_beta_O2", "rel_gamma": "rel_gamma_O2"})
O2_df2 = O2_df2.rename(columns={"exponent": "exponent_O2", "offset": "offset_O2", "adj_delta": "adj_delta_O2", "adj_theta": "adj_theta_O2", "adj_alpha": "adj_alpha_O2", "adj_beta": "adj_beta_O2", "adj_gamma": "adj_gamma_O2"})
F7_df1 = F7_df1.rename(columns={"rel_delta": "rel_delta_F7", "rel_theta": "rel_theta_F7", "rel_alpha": "rel_alpha_F7", "rel_beta": "rel_beta_F7", "rel_gamma": "rel_gamma_F7"})
F7_df2 = F7_df2.rename(columns={"exponent": "exponent_F7", "offset": "offset_F7", "adj_delta": "adj_delta_F7", "adj_theta": "adj_theta_F7", "adj_alpha": "adj_alpha_F7", "adj_beta": "adj_beta_F7", "adj_gamma": "adj_gamma_F7"})
F8_df1 = F8_df1.rename(columns={"rel_delta": "rel_delta_F8", "rel_theta": "rel_theta_F8", "rel_alpha": "rel_alpha_F8", "rel_beta": "rel_beta_F8", "rel_gamma": "rel_gamma_F8"})
F8_df2 = F8_df2.rename(columns={"exponent": "exponent_F8", "offset": "offset_F8", "adj_delta": "adj_delta_F8", "adj_theta": "adj_theta_F8", "adj_alpha": "adj_alpha_F8", "adj_beta": "adj_beta_F8", "adj_gamma": "adj_gamma_F8"})
T3_df1 = T3_df1.rename(columns={"rel_delta": "rel_delta_T3", "rel_theta": "rel_theta_T3", "rel_alpha": "rel_alpha_T3", "rel_beta": "rel_beta_T3", "rel_gamma": "rel_gamma_T3"})
T3_df2 = T3_df2.rename(columns={"exponent": "exponent_T3", "offset": "offset_T3", "adj_delta": "adj_delta_T3", "adj_theta": "adj_theta_T3", "adj_alpha": "adj_alpha_T3", "adj_beta": "adj_beta_T3", "adj_gamma": "adj_gamma_T3"})
T4_df1 = T4_df1.rename(columns={"rel_delta": "rel_delta_T4", "rel_theta": "rel_theta_T4", "rel_alpha": "rel_alpha_T4", "rel_beta": "rel_beta_T4", "rel_gamma": "rel_gamma_T4"})
T4_df2 = T4_df2.rename(columns={"exponent": "exponent_T4", "offset": "offset_T4", "adj_delta": "adj_delta_T4", "adj_theta": "adj_theta_T4", "adj_alpha": "adj_alpha_T4", "adj_beta": "adj_beta_T4", "adj_gamma": "adj_gamma_T4"})
T5_df1 = T5_df1.rename(columns={"rel_delta": "rel_delta_T5", "rel_theta": "rel_theta_T5", "rel_alpha": "rel_alpha_T5", "rel_beta": "rel_beta_T5", "rel_gamma": "rel_gamma_T5"})
T5_df2 = T5_df2.rename(columns={"exponent": "exponent_T5", "offset": "offset_T5", "adj_delta": "adj_delta_T5", "adj_theta": "adj_theta_T5", "adj_alpha": "adj_alpha_T5", "adj_beta": "adj_beta_T5", "adj_gamma": "adj_gamma_T5"})
T6_df1 = T6_df1.rename(columns={"rel_delta": "rel_delta_T6", "rel_theta": "rel_theta_T6", "rel_alpha": "rel_alpha_T6", "rel_beta": "rel_beta_T6", "rel_gamma": "rel_gamma_T6"})
T6_df2 = T6_df2.rename(columns={"exponent": "exponent_T6", "offset": "offset_T6", "adj_delta": "adj_delta_T6", "adj_theta": "adj_theta_T6", "adj_alpha": "adj_alpha_T6", "adj_beta": "adj_beta_T6", "adj_gamma": "adj_gamma_T6"})
Fz_df1 = Fz_df1.rename(columns={"rel_delta": "rel_delta_Fz", "rel_theta": "rel_theta_Fz", "rel_alpha": "rel_alpha_Fz", "rel_beta": "rel_beta_Fz", "rel_gamma": "rel_gamma_Fz"})
Fz_df2 = Fz_df2.rename(columns={"exponent": "exponent_Fz", "offset": "offset_Fz", "adj_delta": "adj_delta_Fz", "adj_theta": "adj_theta_Fz", "adj_alpha": "adj_alpha_Fz", "adj_beta": "adj_beta_Fz", "adj_gamma": "adj_gamma_Fz"})
Cz_df1 = Cz_df1.rename(columns={"rel_delta": "rel_delta_Cz", "rel_theta": "rel_theta_Cz", "rel_alpha": "rel_alpha_Cz", "rel_beta": "rel_beta_Cz", "rel_gamma": "rel_gamma_Cz"})
Cz_df2 = Cz_df2.rename(columns={"exponent": "exponent_Cz", "offset": "offset_Cz", "adj_delta": "adj_delta_Cz", "adj_theta": "adj_theta_Cz", "adj_alpha": "adj_alpha_Cz", "adj_beta": "adj_beta_Cz", "adj_gamma": "adj_gamma_Cz"})
Pz_df1 = Pz_df1.rename(columns={"rel_delta": "rel_delta_Pz", "rel_theta": "rel_theta_Pz", "rel_alpha": "rel_alpha_Pz", "rel_beta": "rel_beta_Pz", "rel_gamma": "rel_gamma_Pz"})
Pz_df2 = Pz_df2.rename(columns={"exponent": "exponent_Pz", "offset": "offset_Pz", "adj_delta": "adj_delta_Pz", "adj_theta": "adj_theta_Pz", "adj_alpha": "adj_alpha_Pz", "adj_beta": "adj_beta_Pz", "adj_gamma": "adj_gamma_Pz"})


#drop 'Group' and 'channel' from all dataframes except Fp1
Fp2_df1 = Fp2_df1.drop(columns=['Group', 'channel'])
F3_df1 = F3_df1.drop(columns=['Group', 'channel'])
F4_df1 = F4_df1.drop(columns=['Group', 'channel'])
C3_df1 = C3_df1.drop(columns=['Group', 'channel'])
C4_df1 = C4_df1.drop(columns=['Group', 'channel'])
P3_df1 = P3_df1.drop(columns=['Group', 'channel'])
P4_df1 = P4_df1.drop(columns=['Group', 'channel'])
O1_df1 = O1_df1.drop(columns=['Group', 'channel'])
O2_df1 = O2_df1.drop(columns=['Group', 'channel'])
F7_df1 = F7_df1.drop(columns=['Group', 'channel'])
F8_df1 = F8_df1.drop(columns=['Group', 'channel'])
T3_df1 = T3_df1.drop(columns=['Group', 'channel'])
T4_df1 = T4_df1.drop(columns=['Group', 'channel'])
T5_df1 = T5_df1.drop(columns=['Group', 'channel'])
T6_df1 = T6_df1.drop(columns=['Group', 'channel'])
Fz_df1 = Fz_df1.drop(columns=['Group', 'channel'])
Cz_df1 = Cz_df1.drop(columns=['Group', 'channel'])
Pz_df1 = Pz_df1.drop(columns=['Group', 'channel'])


Fp2_df2 = Fp2_df2.drop(columns=['Group', 'channel'])
F3_df2 = F3_df2.drop(columns=['Group', 'channel'])
F4_df2 = F4_df2.drop(columns=['Group', 'channel'])
C3_df2 = C3_df2.drop(columns=['Group', 'channel'])
C4_df2 = C4_df2.drop(columns=['Group', 'channel'])
P3_df2 = P3_df2.drop(columns=['Group', 'channel'])
P4_df2 = P4_df2.drop(columns=['Group', 'channel'])
O1_df2 = O1_df2.drop(columns=['Group', 'channel'])
O2_df2 = O2_df2.drop(columns=['Group', 'channel'])
F7_df2 = F7_df2.drop(columns=['Group', 'channel'])
F8_df2 = F8_df2.drop(columns=['Group', 'channel'])
T3_df2 = T3_df2.drop(columns=['Group', 'channel'])
T4_df2 = T4_df2.drop(columns=['Group', 'channel'])
T5_df2 = T5_df2.drop(columns=['Group', 'channel'])
T6_df2 = T6_df2.drop(columns=['Group', 'channel'])
Fz_df2 = Fz_df2.drop(columns=['Group', 'channel'])
Cz_df2 = Cz_df2.drop(columns=['Group', 'channel'])
Pz_df2 = Pz_df2.drop(columns=['Group', 'channel'])


#merge according to 'subject' column

df1 = pd.merge(Fp1_df1, Fp2_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, F3_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, F4_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, C3_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, C4_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, P3_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, P4_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, O1_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, O2_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, F7_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, F8_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, T3_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, T4_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, T5_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, T6_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, Fz_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, Cz_df1, left_on=['subject'], right_on=['subject'], how='inner')
df1 = pd.merge(df1, Pz_df1, left_on=['subject'], right_on=['subject'], how='inner')


df2 = pd.merge(Fp1_df2, Fp2_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, F3_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, F4_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, C3_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, C4_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, P3_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, P4_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, O1_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, O2_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, F7_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, F8_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, T3_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, T4_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, T5_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, T6_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, Fz_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, Cz_df2, left_on=['subject'], right_on=['subject'], how='inner')
df2 = pd.merge(df2, Pz_df2, left_on=['subject'], right_on=['subject'], how='inner')



#check if it all went correctly 
print(df1[0:5])
print(df2[0:5])

#%%

#get categorical output variable and drop it along with the subject label and channel name from the input dataframes
y1 = df1['Group']
y2 = df2['Group']

X1 = df1.drop(columns=['Group', 'subject', 'channel'])
X2 = df2.drop(columns=['Group', 'subject', 'channel'])

#Oversampling of minority class using SMOTE
oversample = SMOTE()
X1_res, y1_res = oversample.fit_resample(X1, y1)
X2_res, y2_res = oversample.fit_resample(X2, y2)

#plot new distribution to check that everything worked correctly:
plt.hist(y1_res, 20)
plt.hist(y2_res, 20)


#5: split into training and test set
x1_tr, x1_te, y1_tr, y1_te = train_test_split(X1_res, y1_res, test_size=0.2, random_state=42, stratify = y1_res)
x2_tr, x2_te, y2_tr, y2_te = train_test_split(X2_res, y2_res, test_size=0.2, random_state=42, stratify = y2_res)


#from checking the df earlier, we know that the predictor variables have very different magnitudes, therefore, let's standardize:
sc = StandardScaler()
x1_tr = sc.fit_transform(x1_tr)
x1_te = sc.transform(x1_te)
x2_tr = sc.fit_transform(x2_tr)
x2_te = sc.transform(x2_te)



#train first classifiers with default settings:
rf1_default = RandomForestClassifier().fit(x1_tr, y1_tr);
rf2_default = RandomForestClassifier().fit(x2_tr, y2_tr);

#accuracy:
print("\t\t\t\t\t\t\t\t\t TR")
print("Random forest relative power acc:\t %.2f\t" % (rf1_default.score(x1_tr, y1_tr)))
print("Random forest aper. adjusted acc:\t %.2f\t" % (rf2_default.score(x2_tr, y2_tr)))


#%%
#Hyperparameter optimization:
def objective(trial):
    rf = RandomForestClassifier()
    param_distributions = {
        # Number of trees:
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log = True),
        # Number of features to consider at every split:
        "max_features": trial.suggest_categorical("max_features", ['log2', 'sqrt', None]),
        # Maximum number of levels in each tree:
        "max_depth": trial.suggest_int("max_depth", 10, 110, log = True),
        # Minimum number of samples required to split a node:
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, log = True),
        # Minimum number of samples required at each leaf node:
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, log = True),
        # Are bootstrap samples or is the whole dataset used when building trees:
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        # Criterion to measure the quality of a split:
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
    }

    cv_score = cross_val_score(rf, x1_tr, y1_tr, n_jobs=4, cv=5)
    mean_cv_accuracy = cv_score.mean()
    return mean_cv_accuracy


pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study()
study.optimize(objective, n_trials=100)

print("================================================")
trial = study.best_trial
print("Best Accuracy relative power model: {:.2f} %".format(trial.value*100))
print("Best Hyperparameters relative power model: {}".format(trial.params))
print("================================================")


def objective(trial):
    rf = RandomForestClassifier()
    param_distributions = {
        # Number of trees:
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log = True),
        # Number of features to consider at every split:
        "max_features": trial.suggest_categorical("max_features", ['log2', 'sqrt', None]),
        # Maximum number of levels in each tree:
        "max_depth": trial.suggest_int("max_depth", 10, 110, log = True),
        # Minimum number of samples required to split a node:
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, log = True),
        # Minimum number of samples required at each leaf node:
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, log = True),
        # Are bootstrap samples or is the whole dataset used when building trees:
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        # Criterion to measure the quality of a split:
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
    }

    cv_score = cross_val_score(rf, x2_tr, y2_tr, n_jobs=4, cv=5)
    mean_cv_accuracy = cv_score.mean()
    return mean_cv_accuracy


pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study()
study.optimize(objective, n_trials=100)

print("================================================")
trial = study.best_trial
print("Best Accuracy aperiodic model: {:.2f} %".format(trial.value*100))
print("Best Hyperparameters aperiodic model: {}".format(trial.params))
print("================================================")


#%%
#apply the best hyperparameters to the random forests classifier (this needs to be manually changed to the output that is printed in the previous step!)
rf1_opt = RandomForestClassifier(n_estimators = 181, max_features = 'log2', max_depth = 60, min_samples_split = 18, min_samples_leaf = 5, bootstrap = False, criterion = 'log_loss')
rf2_opt = RandomForestClassifier(n_estimators = 63, max_features = None, max_depth = 21, min_samples_split = 14, min_samples_leaf = 1, bootstrap = True, criterion = 'entropy')

rf1_opt.fit(x1_tr, y1_tr)
rf2_opt.fit(x2_tr, y2_tr)



#accuracy:
print("\t\t\t\t\t\t\t\t TR \t TE")
print("Relative Power default acc:\t\t %.2f\t %.2f" % (rf1_default.score(x1_tr, y1_tr), rf1_default.score(x1_te, y1_te)))
print("Relative Power optimized acc:\t %.2f\t %.2f" % (rf1_opt.score(x1_tr, y1_tr), rf1_opt.score(x1_te, y1_te)))
print("Aper. Adjusted default acc:\t\t %.2f\t %.2f" % (rf2_default.score(x2_tr, y2_tr), rf2_default.score(x2_te, y2_te)))
print("Aper. Adjusted optimized acc:\t %.2f\t %.2f" % (rf2_opt.score(x2_tr, y2_tr), rf2_opt.score(x2_te, y2_te)))


#%%
#confusion matrices:
font = {'size'   : 20}
plt.rc('font', **font)

y1_pred_d = rf1_opt.predict(x1_te)
cmd = confusion_matrix(y1_te, y1_pred_d)
cmd_display = ConfusionMatrixDisplay(cmd, display_labels=np.array(['AD', 'FTD', 'HC'])).plot()
cmd_display.ax_.set_title('Confusion Matrix Relative Power')
plt.show()

y2_pred_o = rf2_opt.predict(x2_te)
cmo = confusion_matrix(y2_te, y2_pred_o)
cmo_display = ConfusionMatrixDisplay(cmo, display_labels=np.array(['AD', 'FTD', 'HC'])).plot()
cmo_display.ax_.set_title('Confusion Matrix Aperiodic Adjusted')
plt.show()


#%%
#calculate accuracy, sensitivity, and specificity from the number of correctly classified cases (can be read from the confusion matrices)
acc_pow = (2+4+6)/22
acc_ape = (4+6+7)/22

#sens = TP / TP+FN
sens1_AD = 2 / (3+2+2)
sens1_FTD = 4 / (4+2+1)
sens1_HC = 6 / (6+1+1)
#spec = TN / TN+FP
spec1_AD = (4+6+2+1)/(4+6+2+1+1+1)
spec1_FTD = (2+1+6+2)/(2+1+6+2+3+1)
spec1_HC = (4+1+2+3)/(4+1+2+3+2+2)

#sens = TP / TP+FN
sens_AD = 4 / (7)
sens_FTD = 6/7
sens_HC = 7/8
#spec = TN / TN+FP
spec_AD = (6+7)/(6+7+1+1)
spec_FTD = (4+7+1+1)/(4+7+1+1+2+0)
spec_HC = (6+4+2+1)/(6+4+2+1+1+0)


print("Accuracy Power:", acc_pow)
print("Accuracy Aperiodic:", acc_ape)

print("Sensitivity Power AD:", sens1_AD)
print("Sensitivity Power FTD:", sens1_FTD)
print("Sensitivity Power HC:", sens1_HC)

print("Specificity Power AD:", spec1_AD)
print("Specificity Power FTD:", spec1_FTD)
print("Specificity Power HC:", spec1_HC)

print("Sensitivity Aperiodic AD:", sens_AD)
print("Sensitivity Aperiodic FTD:", sens_FTD)
print("Sensitivity Aperiodic HC:", sens_HC)

print("Specificity Aperiodic AD:", spec_AD)
print("Specificity Aperiodic FTD:", spec_FTD)
print("Specificity Aperiodic HC:", spec_HC)


#%%
#significance testing with McNemar's:

#make dataframe of true and predicted labels, along with a column showing whether each classifier is correct or not
d1 = {'True_labels': y1_te, 'Pred_RelPow': y1_pred_d, 'Pred_AperAdj': y2_pred_o}
df_for_sig = pd.DataFrame(data = d1)
df_for_sig = df_for_sig.reset_index(drop = True)

for i, j in df_for_sig.iterrows():
  if j['True_labels'] == j['Pred_RelPow']:
    df_for_sig.loc[i, 'RelPow_Corr'] = True
  else:
    df_for_sig.loc[i, 'RelPow_Corr'] = False
  if j['True_labels'] == j['Pred_AperAdj']:
    df_for_sig.loc[i, 'AperAdj_Corr'] = True
  else:
    df_for_sig.loc[i, 'AperAdj_Corr'] = False
    

FTD_df = df_for_sig[df_for_sig["True_labels"] == 'F']
AD_df = df_for_sig[df_for_sig["True_labels"] == 'A']
HC_df = df_for_sig[df_for_sig["True_labels"] == 'H']

for i, j in FTD_df.iterrows():
  if j['True_labels'] == j['Pred_RelPow']:
    FTD_df.loc[i, 'RelPow_Corr'] = True
  else:
    FTD_df.loc[i, 'RelPow_Corr'] = False
  if j['True_labels'] == j['Pred_AperAdj']:
    FTD_df.loc[i, 'AperAdj_Corr'] = True
  else:
    FTD_df.loc[i, 'AperAdj_Corr'] = False

for i, j in AD_df.iterrows():
  if j['True_labels'] == j['Pred_RelPow']:
    AD_df.loc[i, 'RelPow_Corr'] = True
  else:
    AD_df.loc[i, 'RelPow_Corr'] = False
  if j['True_labels'] == j['Pred_AperAdj']:
    AD_df.loc[i, 'AperAdj_Corr'] = True
  else:
    AD_df.loc[i, 'AperAdj_Corr'] = False

for i, j in HC_df.iterrows():
  if j['True_labels'] == j['Pred_RelPow']:
    HC_df.loc[i, 'RelPow_Corr'] = True
  else:
    HC_df.loc[i, 'RelPow_Corr'] = False
  if j['True_labels'] == j['Pred_AperAdj']:
    HC_df.loc[i, 'AperAdj_Corr'] = True
  else:
    HC_df.loc[i, 'AperAdj_Corr'] = False


#Get number of cases in which classifier 1 and 2 are both correct, only one or the other, or both incorrect
Cl1Corr_Cl2Corr = 0
Cl1Corr_Cl2NCorr = 0
Cl1NCorr_Cl2Corr = 0
Cl1NCorr_Cl2NCorr = 0


#here, McNemar's test is run for the whole sample. To repeat this for the class-specific tests, replace "df_for_sig" in the first line with
#AD_df, FTD_df, and HC_df

for i, j in df_for_sig.iterrows():
  if j['RelPow_Corr'] == True and j['AperAdj_Corr'] == True:
    Cl1Corr_Cl2Corr = Cl1Corr_Cl2Corr + 1
  elif j['RelPow_Corr'] == True and j['AperAdj_Corr'] == False:
    Cl1Corr_Cl2NCorr = Cl1Corr_Cl2NCorr + 1
  elif j['RelPow_Corr'] == False and j['AperAdj_Corr'] == True:
    Cl1NCorr_Cl2Corr = Cl1NCorr_Cl2Corr + 1
  elif j['RelPow_Corr'] == False and j['AperAdj_Corr'] == False:
    Cl1NCorr_Cl2NCorr = Cl1NCorr_Cl2NCorr + 1

from statsmodels.stats.contingency_tables import mcnemar
# define contingency table
table = [[Cl1Corr_Cl2Corr, Cl1Corr_Cl2NCorr],
 [Cl1NCorr_Cl2Corr, Cl1NCorr_Cl2NCorr]]

# calculate mcnemar's:
result = mcnemar(table, exact=False)

print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

alpha = 0.05
if result.pvalue > alpha:
 print('Same proportions of errors (fail to reject H0)')
else:
 print('Different proportions of errors (reject H0)')



#%%
##Plot one-vs-rest ROC curves:
#from sklearn import metrics

font = {'size'   : 15}
plt.rc('font', **font)

label_binarizer = LabelBinarizer().fit(y1_tr)
y_onehot_test = label_binarizer.transform(y1_te)
y_score = rf1_opt.fit(x1_tr, y1_tr).predict_proba(x1_te)

label_binarizer2 = LabelBinarizer().fit(y2_tr)
y_onehot_test2 = label_binarizer2.transform(y2_te)
y_score2 = rf2_opt.fit(x2_tr, y2_tr).predict_proba(x2_te)

class_of_interest1 = "A"
class_id1 = np.flatnonzero(label_binarizer.classes_ == class_of_interest1)[0]
class_of_interest2 = "F"
class_id2 = np.flatnonzero(label_binarizer.classes_ == class_of_interest2)[0]
class_of_interest3 = "C"
class_id3 = np.flatnonzero(label_binarizer.classes_ == class_of_interest3)[0]

class_id1_2 = np.flatnonzero(label_binarizer2.classes_ == class_of_interest1)[0]
class_id2_2 = np.flatnonzero(label_binarizer2.classes_ == class_of_interest2)[0]
class_id3_2 = np.flatnonzero(label_binarizer2.classes_ == class_of_interest3)[0]


fpr, tpr, roc_auc = dict(), dict(), dict()
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

fpr2, tpr2, roc_auc2 = dict(), dict(), dict()
fpr2["micro"], tpr2["micro"], _ = roc_curve(y_onehot_test2.ravel(), y_score2.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])


font = {'size'   : 12}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(8, 6))

plt.plot([0, 1], [0, 1], linestyle='dashdot', color='gray', label='Chance level')


RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id1],
        y_score[:, class_id1],
        name="RelPow AD vs Rest",
        color="#ff7d33",
        linestyle='dashed',
        ax=ax
    )

RocCurveDisplay.from_predictions(
        y_onehot_test2[:, class_id1],
        y_score2[:, class_id1],
        name="AperAdj AD vs Rest",
        color="#ff7d33",
        ax=ax
    )

RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id2],
        y_score[:, class_id2],
        name="RelPow FTD vs Rest",
        color="#6cdfae",
        linestyle='dashed',
        ax=ax
    )

RocCurveDisplay.from_predictions(
        y_onehot_test2[:, class_id2],
        y_score2[:, class_id2],
        name="AperAdj FTD vs Rest",
        color="#6cdfae",
        ax=ax
    )


RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id3],
        y_score[:, class_id3],
        name="RelPow HC vs Rest",
        color="#79aff4",
        linestyle='dashed',
        ax=ax
    )


RocCurveDisplay.from_predictions(
        y_onehot_test2[:, class_id3],
        y_score2[:, class_id3],
        name="AperAdj HC vs Rest",
        color="#79aff4",
        ax=ax
    )

_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves:"
)





