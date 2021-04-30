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

import urllib.request
import zipfile
from pycocotools.coco import COCO
from nsd_access import NSDAccess

from IPython import embed


class NSDIAccess(NSDAccess):
    """
    A derived class from NSDAccess that provides easy access to the NSD Imagery data
    """
    def __init__(self, nsd_folder, *args, **kwargs):
        super().__init__(nsd_folder, *args, **kwargs)
        self.stimuli_file = op.join(
            self.nsd_folder, 'nsddata_stimuli', 'stimuli', 'nsdimagery', 
            'allstim.mat') # check about this
        self.behavior_file = op.join(os.getcwd(), 'meta_data', '{subject}_Imagery_Behavioralresponse.csv')
        
    def read_betas(self, subject, trial_index=[], data_type='nsdimagerybetas_fithrf', data_format='fsaverage', mask=None):
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
            ipf = op.join(data_folder,'betas_nsdimagery.mat')
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
                    data_folder, f'{hemi}.betas_nsdimagery.mgh')).get_data()
                session_betas.append(hdata)
            out_data = np.squeeze(np.vstack(session_betas))
        else:
            # if no mask was specified, we'll use the nifti image
            out_data = nb.load(
                op.join(data_folder, 'betas_nsdimagery.nii.gz')).get_data()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]
    
    def get_expinfo(self):
        """ Get experiment information
        
        """
        df = pd.read_csv(op.join(os.getcwd(), "meta_data", "nsdimagery_stiminfo.csv"))
        return df
                         
    def read_behavior(self, subject, expinfo = False):
        """read_behavior [summary]

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        expinfo : bool
            Should the experiment info concatenated with the behaviour responses
            
        Returns
        -------
        pandas DataFrame
            DataFrame containing the behavioral information
        """

        behavior = pd.read_csv(self.behavior_file.format(
            subject=subject))

        if expinfo:
            df1 = self.get_expinfo()
            result = pd.concat([df1, behavior], axis = 1)
        else:
            result = behavior

        return result

    def read_images(self, image_index, show=False):
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
        sdataset = sf.get('images')
        f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6*len(image_index), 6))
        if len(image_index) < 2:
            ss = [ss]
        if show == True:
            for a in range(len(image_index)):
                ss[a].axis('off')
                ss[a].imshow(np.transpose(sdataset[image_index[a]],(1, 2, 0)))
        val = np.empty(sdataset[0].shape)       
        for a in image_index:
            if len(val) == 0:
                val = sdataset[a]
                val = np.concatenate(val, sdataset[a])
        return val
    
    