#!/usr/bin/env python
# coding: utf-8


import os
import os.path as op
import glob
import nibabel as nb
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
import urllib.request
import zipfile
from nsd_access import NSDAccess

from IPython import embed



class NSDSAccess(NSDAccess):
    """
    A derived class from NSDAccess that provides easy access to the NSD Synthetic data
    """
    def __init__(self, nsd_folder, *args, **kwargs):
        super().__init__(nsd_folder, *args, **kwargs)
        self.stimuli_file = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsdsynthetic', 
            'nsdsynthetic_stimuli.hdf5')
        self.stimuli_file_color = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsdsynthetic', 
            'nsdsynthetic_colorstimuli_{}.hdf5')
        self.stim_index_file = op.join(
            self.nsd_folder, 'nsddata','experiments','nsdsynthetic',
            'nsdsynthetic_expdesign.mat')
        # To read in the behavior responses set the path here, after the behavior files are generated
        
    def read_betas(self, subject, trial_index=[], data_type='nsdsyntheticbetas_fithrf_GLMdenoise_RR', data_format='fsaverage', mask=None):
        """read_betas read betas from MRI files
        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        trial_index : list, optional
            which trials from this session's file to return, by default [], which returns all trials
        data_type : str, optional
            which type of beta values to return from ['betas_assumehrf', 'betas_fithrf', 'betas_fithrf_GLMdenoise_RR', 'restingbetas_fithrf'], by default 'betas_fithrf_GLMdenoise_RR'
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm'], by default 'fsaverage'
        mask : numpy.ndarray, if defined, selects 'mat' data_format, needs volumetric data_format
            binary/boolean mask into mat file beta data format.
        Returns
        -------
        numpy.ndarray, 2D (fsaverage) or 4D (other data formats)
            the requested per-trial beta values
        """
        data_folder = op.join(self.nsddata_betas_folder,
                              subject, data_format, data_type)

        if type(mask) == np.ndarray:  # will use the mat file iff exists, otherwise boom!
            ipf = op.join(data_folder,'betas_nsdsynthetic.mat')
            assert op.isfile(ipf),                 'Error: ' + ipf + ' not available for masking. You may need to download these separately.'
            # will do indexing of both space and time in one go for this option,
            # so will return results immediately from this
            h5 = h5py.File(ipf, 'r')
            betas = h5.get('betas')
            # embed()
            if len(trial_index) == 0:
                trial_index = slice(0, betas.shape[0])
            # this isn't finished yet - binary masks cannot be used for indexing like this
            return betas[trial_index, np.nonzero(mask)]

        if data_format == 'fsaverage':
            session_betas = []
            for hemi in ['lh', 'rh']:
                hdata = nb.load(op.join(
                    data_folder, f'{hemi}.betas_nsdsynthetic.mgh')).get_data()
                session_betas.append(hdata)
            out_data = np.squeeze(np.vstack(session_betas))
        else:
            # if no mask was specified, we'll use the nifti image
            out_data = nb.load(
                op.join(data_folder, 'betas_nsdsynthetic.nii.gz')).get_data()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]
    
    def stim_indexing(self):
        '''
        A function to return the stimulus indexing information for the betas in the session
        '''
        mat_contents = sio.loadmat(self.stim_index_file)
        df = pd.DataFrame(mat_contents['masterordering'], index = ['ImageIndex']).T
        d = np.diff(mat_contents['masterordering'])
        d2 = np.where(d == 0)
        req = np.append(d2[1],d2[1] + 1)
        df['OnebackTrial'] = -1
        df['TaskType'] = -1
        for a in range(len(df)):
            if a in req:
                df['OnebackTrial'][a] = 1
            else:
                df['OnebackTrial'][a] = 0
        count = 0

        for i in range(len(mat_contents['stimpattern'][0])):
            for j in range(len(mat_contents['stimpattern'][0][0])):
                if mat_contents['stimpattern'][0][i][j] == 1: # Stim trial
                    if i % 2 == 0: #  Fixation [0,2,4,6]
                        df['TaskType'][count] = 0
                
                    else: # Memory [1,3,5,7]
                        df['TaskType'][count] = 1
                    count += 1
        return df
    
    def read_images(self, subject, image_index, show=False):
        """read_images reads a list of images, and returns their data
        Parameters
        ----------
        image_index : list of integers
            which images indexed in the 73k format to return
        show : bool, optional
            whether to also show the images, by default False
        Returns
        -------
        numpy.ndarray, 3D
            RGB image data
        """

        sf = h5py.File(self.stimuli_file, 'r')
        sdataset = sf.get('imgBrick')
        if max(image_index) > 219:
            sf2 = sf = h5py.File(self.stimuli_file_color.format(subject), 'r')
            sdataset2 = sf2.get('imgBrick')
        siz = list(sdataset[0].shape)  
        siz.insert(0, len(image_index))
        vals = np.empty(siz, dtype=int)    
        for a in range(len(image_index)):
            if image_index[a] < 220:
                vals[a] = sdataset[image_index[a]]
            else:
                vals[a] = sdataset2[image_index[a]-220]

        f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6*len(image_index), 6))
        if len(image_index) < 2:
            ss = [ss]
        if show == True:
            for a in range(len(image_index)):
                ss[a].axis('off')
                ss[a].imshow(vals[a])
        return vals
        




