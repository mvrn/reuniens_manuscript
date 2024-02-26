import os
from posix import listdir
from numpy.core.defchararray import startswith
import pandas as pd
import numpy as np
import nibabel as nb
import scipy.stats as stats
from glob import glob
from copy import deepcopy
%matplotlib inline
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statannot


proj_dir = '/home/data/madlab/McMakin_EMU/derivatives'

subjects = os.listdir(f'{proj_dir}/dwi/mvrn/mvrn_preed_thal_hemikmeans/')

mask_filenames = {}
cope_test = {}
cope_study = {}

# Iterating over subjects to get their respective contrasts based on our masks of interest (RE, CA1, mPFC)
for sid in subjects:

    # Defining path to STUDY RE, CA1, and mPFC masks (in subject-specific space)
    mask_filenames[sid]= glob(f'{proj_dir}/masks/mvrn/new_midthal/new_RE_inepi/study/{sid}/temp2epi/_subject_id_{sid}/*.nii')
    mask_filenames[sid].extend(glob(f'{proj_dir}/masks/mvrn/new_midthal/CA1_inepi/study/{sid}/temp2epi/_subject_id_{sid}/*.nii.gz'))
    mask_filenames[sid].extend(glob(f'{proj_dir}/masks/mvrn/new_midthal/mPFC_inepi/study/{sid}/temp2epi/_subject_id_{sid}/*.nii.gz'))

    # Defining path to TEST RE, CA1, and mPFC masks (in subject-specific space)
    mask_filenames[sid].extend(glob(f'{proj_dir}/masks/mvrn/new_midthal/new_RE_inepi/test/{sid}/temp2epi/_subject_id_{sid}/*.nii'))
    mask_filenames[sid].extend(glob(f'{proj_dir}/masks/mvrn/new_midthal/CA1_inepi/test/{sid}/temp2epi/_subject_id_{sid}/*.nii.gz'))
    mask_filenames[sid].extend(glob(f'{proj_dir}/masks/mvrn/new_midthal/mPFC_inepi/test/{sid}/temp2epi/_subject_id_{sid}/*.nii.gz'))

    # Getting the STUDY & TEST BSC contrats per subject
    cope_study[sid] = sorted(glob(f'{proj_dir}/frstlvl/study_lss_betaseries/{sid}/*_?????_*.nii.gz'))
    cope_test[sid] = sorted(glob(f'{proj_dir}/frstlvl/test_lss_betaseries/{sid}/Lures*_????.nii.gz'))

    # Checking that I'm capturing all my masks and copes
    # print(mask_filenames[sid])
    
    # print(cope_study[sid])

print('Dict Complete') 

all_data = {}
# subjects = ['1001']

for sid in subjects:

    print(sid, 'Mask')

    curr_subj_dict = {}
    subj_mask = {}
    copes = {}

    for mask in mask_filenames[sid]:
        name, _ = os.path.basename(mask).split('_trans')
      
        # Loading matrices with our mask data
        subj_mask[name] = nb.load(mask).get_fdata()[:,:,:,0]
        
    # Splitting copes labels so as to get relevant portion (e.g., LureCRNeg_run1)

    # Study
    for cp_std in cope_study[sid]:
        name, _, _ = os.path.basename(cp_std).split('.')

        # Loading matrices with our cope study data
        copes[name] = nb.load(cp_std).get_fdata()

    # Test
    for cp_tst in cope_test[sid]:
        name, _, _ = os.path.basename(cp_tst).split('.')

        # Loading matrices with our cope test data
        copes[name] = nb.load(cp_tst).get_fdata()

    # Getting contrast labels only (without run number) for study and test
    for session in ['study', 'test']:
        if session == 'study':
            # Listing STUDY contrasts for negative, neutral, and positive stimuli
            orig_study_contrasts = os.listdir(f'{proj_dir}/frstlvl/{session}_lss_betaseries/{sid}')
            # Removing the run and nii.gz parts of the contrast labels
            split_study_contrasts = [x[:-12] for x in orig_study_contrasts]

        elif session == 'test':
            # Listing TEST contrasts for negative, neutral, and positive stimuli
            orig_test_contrasts = os.listdir(f'{proj_dir}/frstlvl/{session}_lss_betaseries/{sid}')

            # Removing the run and nii.gz parts of the contrast labels
            split_test_contrasts = [x[:-12] for x in orig_test_contrasts]
          
            # Adding study contratst to dict. 

            for cont_study in split_study_contrasts:

                if f'{cont_study}_run1' in copes.keys() and f'{cont_study}_run2' not in copes.keys():
                    copes[f'{cont_study}_combined'] = deepcopy(copes[f'{cont_study}_run1'])
                    
                elif f'{cont_study}_run2' in copes.keys() and f'{cont_study}_run1' not in copes.keys():
                    copes[f'{cont_study}_combined'] = deepcopy(copes[f'{cont_study}_run2'])
                
                elif f'{cont_study}_run1' in copes.keys() and f'{cont_study}_run2' in copes.keys():
                    copes[f'{cont_study}_combined'] = np.concatenate((copes[f'{cont_study}_run1'], copes[f'{cont_study}_run2']), axis=3)  
                 

            for cont_test in split_test_contrasts:
                
                # original attempt
                if f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run2' not in copes.keys() and f'{cont_test}_run3' not in copes.keys() and f'{cont_test}_run4' not in copes.keys():
                    copes[f'{cont_test}_combined'] = deepcopy(copes[f'{cont_test}_run1'])
                    
                elif f'{cont_test}_run2' in copes.keys() and f'{cont_test}_run1' not in copes.keys() and f'{cont_test}_run3' not in copes.keys() and f'{cont_test}_run4' not in copes.keys():
                    copes[f'{cont_test}_combined'] = deepcopy(copes[f'{cont_test}_run2'])
                
                elif f'{cont_test}_run3' in copes.keys() and f'{cont_test}_run1' not in copes.keys() and f'{cont_test}_run2' not in copes.keys() and f'{cont_test}_run4' not in copes.keys():
                    copes[f'{cont_test}_combined'] = deepcopy(copes[f'{cont_test}_run3'])
                
                elif f'{cont_test}_run4' in copes.keys() and f'{cont_test}_run1' not in copes.keys() and f'{cont_test}_run2' not in copes.keys() and f'{cont_test}_run3' not in copes.keys():
                    copes[f'{cont_test}_combined'] = deepcopy(copes[f'{cont_test}_run4'])

                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run2' in copes.keys() and f'{cont_test}_run3' in copes.keys() and f'{cont_test}_run4' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run2'], copes[f'{cont_test}_run3'], copes[f'{cont_test}_run4']), axis=3)  


                # Adding combination of available contrast parameter-runs for subjects that do not meet the criteria from above
                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run2' in copes.keys() and f'{cont_test}_run4' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run2'], copes[f'{cont_test}_run4']), axis=3)

                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run3' in copes.keys() and f'{cont_test}_run4' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run3'], copes[f'{cont_test}_run4']), axis=3)  

                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run4' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run4']), axis=3)  

                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run2' in copes.keys() and f'{cont_test}_run3' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run2'], copes[f'{cont_test}_run3']), axis=3)
            
                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run3' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run3']), axis=3)  

                elif f'{cont_test}_run2' in copes.keys() and f'{cont_test}_run3' in copes.keys() and f'{cont_test}_run4' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run2'], copes[f'{cont_test}_run3'], copes[f'{cont_test}_run4']), axis=3)
    
                elif f'{cont_test}_run2' in copes.keys() and f'{cont_test}_run4' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run2'], copes[f'{cont_test}_run4']), axis=3)
    
                elif f'{cont_test}_run1' in copes.keys() and f'{cont_test}_run2' in copes.keys():
                    copes[f'{cont_test}_combined'] = np.concatenate((copes[f'{cont_test}_run1'], copes[f'{cont_test}_run2']), axis=3)
    
            # printing masks results for each available cope per subject 
            for mask, mask_data in subj_mask.items():

                for cope, contrast_data in copes.items():
                    
                    curr_subj_dict[f'{mask}_{cope}'] = contrast_data[mask_data > 0].mean(axis=(0))

        all_data[sid] = curr_subj_dict

area_list = []
contrast_list = []

# Creates a list of unique regions and contrasts
for x in all_data:
    # Creating copy of dictionary so that it gets updated with the new names
    # for the mask+copes.
    orig_columns = all_data[x].copy().keys()

    for y in orig_columns:
        # Renaming masks+copes labels, such that we are separating
        # the brain area from the copes name
        if y.startswith('midthal'):
            area,contrast = y.split('_vent_')
            # print(area, contrast)

        elif y.startswith('emu_CA1'):
            new_name = y[4:]
            # print(new_name)
            area, contrast = new_name.split('_sf_')
            # print(area, contrast)
        
        elif y.startswith('emu_mvrn'):
            new_name = y[9:]
            temp_name = new_name.replace('mpfc_', 'mpfc_temp_')
            area, contrast = temp_name.split('_temp_')
            # print(area,contrast)
        all_data[x][f'{area}-{contrast}'] = all_data[x][y]

        area_list.append(area)
        contrast_list.append(contrast)

area_list = list(set(area_list))
# print(area_list)
contrast_list = list(set(contrast_list))

stat_dict = {'Subject':[]}

for contrast in contrast_list:
    for area in area_list:
        # Checking to see if there are no areas
        for area_2 in [x for x in area_list if area not in x]:
            stat_dict[f'{area}_{area_2}-{contrast}'] = []

#Calculates Pearson R values for each regionXregionXcontrast pairing or returns a nan 
#if a regionXcontrast is missing or if a contrast has less than 4 trials
for key in sorted(all_data):
    stat_dict['Subject'].append(key)
   
    for contrast in contrast_list:
        for area in area_list:

        # Checking to see if there are no areas
            for area_2 in [x for x in area_list if area not in x]:
                try:
                    if len(all_data[key][f'{area}-{contrast}']) > 4:
                        stat_dict[f'{area}_{area_2}-{contrast}'].append(\
                        stats.pearsonr(all_data[key][f'{area}-{contrast}'], 
                                       all_data[key][f'{area_2}-{contrast}'])[0])

                    else: 
                        stat_dict[f'{area}_{area_2}-{contrast}'].append(np.nan)

                except KeyError:
                    stat_dict[f'{area}_{area_2}-{contrast}'].append(np.nan)

stat_frame = pd.DataFrame.from_dict(stat_dict)
