import pandas as pd
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from plotnine import *
from src.functions_utils import *


## Defining paths

currdir= Path(os.path.join('..', os.getcwd()))
path = str(currdir.parent)


### Creating Figures directory

figpath = os.path.join(path, 'Figures')

def makedirectory():
    
    if not os.path.exists(figpath):
        
        print("path doesnot exist")
        
        figout = os.makedirs(figpath)
        
        print("Figure directory is created")
        
        return figout
    
    else:
        
        print("Directory already existed")
        
        return None

    
figout = makedirectory()


## Loading Data files

class load_data:
    
    
    def __init__(self,path):
        
        self.path = path
        
        
    def patients_data(self):
        
        lst = [d for d, subdir, files in os.walk(self.path, topdown=True)]

        filepath = [x for x in lst if len(x) == max([len(x) for x in lst])]

        return filepath
    
    

## Calling functions
    
        
subclass = load_data(path)   

patients_data = subclass.patients_data()


## Patient data

### First performing data cleaning and import call files belong to Pre stimulus and Post stimulus Session


def merged_patient_data(patients_data):
    
    
    ''' patient_data: list of folder path'''

    
    merged_data= []
    
    for p in patients_data:
        
        prf = filtering_calfiles(p)
    
        pf = data_extraction(prf)[0]

        ## Extracting Patient folder name

        pf['Patient'] = pf.apply(get_foldernames, axis=1)

        psd = PSD_calculation(pf)

        merged_data.append(psd)

    
    merged_df = pd.concat(merged_data)
    
    return merged_df


merged_df = merged_patient_data(patients_data)


## Plotting patient individually

def plotting_patient_individually(merged_df):
    
    for i, p in enumerate (merged_df.Patient.unique()):

        tp = merged_df[merged_df['Patient'] == str(p)]

        pre_ST_idx = tp.query("type == '124'")['PSD_mean'].idxmax()

        post_ST_idx = tp.query("type == '125'")['PSD_mean'].idxmax()

        preST_peak = tp['freq'][pre_ST_idx]

        postST_peak = tp['freq'][post_ST_idx]

        patient_folder = tp['Patient'][0]   

            # Plotting

        g = (ggplot(tp, aes(x='freq', y='PSD_mean', color='ST')) 
             + geom_line()
             + theme_bw() 
             + theme(figure_size=(8, 6))
             + geom_vline(xintercept=[preST_peak, postST_peak],
                         colour=['#3690c0', '#ef3b2c'],
                         size=[0.8, 1],
                         linetype="dotted")
             + annotate("text", label = "Pre-ST:  " + str(preST_peak) + 'HZ', color="#3690c0", size = 12, x = 9, y = 7)
             + annotate("text", label = "Post-ST:  " + str(postST_peak) + 'HZ', color="#ef3b2c", size = 12, x = 9, y = 6)
             + labs(x='Frequency (Hz)', y='Power Spectral Density (DBM/Hz)')
             + ggtitle("Hand Tremor data: " + patient_folder) 
             + scale_colour_manual(names=['Pre-S', 'Post-ST'],
                                  values=["#ef3b2c","#3690c0"])
             + guides(color=guide_legend(title="Stimulus Session"))

            )

        ggsave(filename='patient_folder_'+str(i) + '.png', plot = g, path = figpath)


    
    
    return 
    

    
plotting_patient_individually(merged_df)
        
    
def plotting_patient_all(merged_df):
    
    ## aggregating Power for all patient
    
    tp = (merged_df.groupby(['freq', 'type', 'ST'])
           .agg({'PSD_mean': 'mean'})
           .reset_index()
           
          )

    pre_ST_idx = tp.query("type == '124'")['PSD_mean'].idxmax()

    post_ST_idx = tp.query("type == '125'")['PSD_mean'].idxmax()

    preST_peak = tp['freq'][pre_ST_idx]

    postST_peak = tp['freq'][post_ST_idx]
 

        # Plotting

    g = (ggplot(tp, aes(x='freq', y='PSD_mean', color='ST')) 
         + geom_line()
         + theme_bw() 
         + theme(figure_size=(8, 6))
         + geom_vline(xintercept=[preST_peak, postST_peak],
                     colour=['#3690c0', '#ef3b2c'],
                     size=[0.8, 1],
                     linetype="dotted")
         + annotate("text", label = "Pre-ST:  " + str(preST_peak) + 'HZ', color="#3690c0", size = 12, x = 9, y = 7)
         + annotate("text", label = "Post-ST:  " + str(postST_peak) + 'HZ', color="#ef3b2c", size = 12, x = 9, y = 6)
         + labs(x='Frequency (Hz)', y='Power Spectral Density (DBM/Hz)')
         + ggtitle("Hand Tremor data of all three Patients") 
         + scale_colour_manual(names=['Pre-S', 'Post-ST'],
                              values=["#ef3b2c","#3690c0"])
         + guides(color=guide_legend(title="Stimulus Session"))

        )

    ggsave(filename='Mean_Patient_tremor_power.png', plot = g, path = figpath)


    
    
    return 
    

    
plotting_patient_all(merged_df)
        
    
    
    

