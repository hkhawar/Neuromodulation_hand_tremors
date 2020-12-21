import pandas as pd
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from plotnine import *



def filtering_calfiles(p):
    

    timestamp=[]

    filenames=[]

    for f in os.listdir(p):

        if f.startswith("cal_") and f.endswith(".csv"):

            calpath = os.path.join(p, f)

            with open(calpath) as csvfile:

                call_data = list(csv.reader(csvfile))

                # Getting index if it matches with string list 

                t_stamp = call_data[5][0]

                timestamp.append(t_stamp)

                filenames.append(os.path.join(p, f))


    cal_time = pd.DataFrame(zip(timestamp, filenames), columns=['timestamp', 'filename'])

    ## reading dev file in each patient

    devpath = [os.path.join(p, d) for d in os.listdir(p) if "dev" in d and d.endswith(".csv")][0]

    ## Processing dev files to get timestamps to match get pre-ST and post-ST cal files

    dev = pd.read_csv(devpath)


    row = (dev.iloc[:, 0] == 'time').idxmax()

    dev.columns = dev.loc[row]

    dev = (dev.loc[row+1:, :]

           .reset_index(drop=True)
          )


    data_dype = ['124', '125']

    dev = (dev[dev['type']

               .isin(data_dype)]

           .sort_values(by=['type'])

           .reset_index(drop=True)

           .rename(columns={'time': 'timestamp'})

          )



    dev["ST"] = dev["type"].map(lambda x: "Pre_ST" if "124" in x else "Post_ST")


    tmp = pd.merge(dev, cal_time, how='inner', on='timestamp')

#     ## Extracting date informationn from stamps

    tmp['Date_(local)'] = (tmp['timestamp']

              .apply(lambda x: datetime.fromtimestamp(int(x))      
              .strftime('%d-%b-%Y'))
              .astype('str')
             )

    tmp['time_s'] = (tmp['timestamp']

              .apply(lambda x: datetime.fromtimestamp(int(x))      
              .strftime("%H:%M:%S"))
              .astype('str')
                    )

    ## Path of Summary file

    spath = [os.path.join(p, s) for s in os.listdir(p) if "Summary" in s and s.endswith(".csv")][0]


    summary = pd.read_csv(spath)


    ## Merging data with summary file


    prf = (pd.merge(summary, tmp, how='inner', on='Date_(local)')
           .drop_duplicates(subset=['timestamp'])

          )

    return prf
    


def data_extraction(tmp):
    
    ## list of paths of cal files used for PRE-ST and Post-ST
    
    cal_flist = list(tmp['filename'])


    fft_data = []
    
    gryo_data = []
    
    acc_data = []

    for i, cal in enumerate(cal_flist):


        match1=['freq', 'fft']
        match2= ['time', 'gyroX', 'gyroY', 'gyroZ']
        match3= ['time', 'accelX', 'accelY', 'accelZ']


        with open(cal) as csvfile:

            call_data = list(csv.reader(csvfile))
            
            # Getting index if it matches with string list 

            index1 = [index for index, f in enumerate(call_data) if f == match1][0]

            index2 = [index for index, f in enumerate(call_data) if f == match2][0]

            index3 = [index for index, f in enumerate(call_data) if f == match3][0]

            timestamp = call_data[5][0]


        fft = pd.DataFrame(call_data[index1+1:index2], columns=match1)       

        gyro = pd.DataFrame(call_data[index2+1:index3], columns=match2)

        acc = pd.DataFrame(call_data[index3+1:], columns=match3) 

        fft['timestamp'] = call_data[5][0]
        gyro['timestamp'] = call_data[5][0]
        acc['timestamp'] = call_data[5][0]

        fft_data.append(fft)
        gryo_data.append(gyro)
        acc_data.append(acc)


    fft = (pd.concat(fft_data)
           .merge(tmp,
                    how='inner', on='timestamp')
          )
        
        
    gryo = (pd.concat(gryo_data)
            .merge(tmp,
                    how='inner', on='timestamp')
           )
    
    acc = (pd.concat(acc_data)
           .merge(tmp,
                    how='inner', on='timestamp')
          )
    
    
    return fft, gryo, acc


def PSD_calculation(fft):
    

    pf=fft.copy()
    
    ## converting columns to floats

    pf[['freq', 'fft']] = (pf[['freq', 'fft']]
              .astype('float')
              .round(2)

             )
    
    
    ## Filtering data which are between 4.0 to 12.03 HZ

    pf = (pf.query('freq >= 4.0 & freq <=12.03')
          
          .reset_index(drop=True)
          
         )
    
    # Calculation PSD from fft values by applying formula
    
    
    # Total no of Samples

    N=pf.shape[0]


    pf['PSD'] = (pf['fft']
                 
                 .apply(lambda x: (1/(2*3.14*N)) * abs(x)** 2)
                 
                )

    # Aggregation of rows grouped by frequency and stimulus session

    tp = (pf.groupby(['freq', 'type', 'ST', 'Patient'])
          
          .agg({'PSD': ['mean','median']})
          
          .reset_index()
          
         )
    
    tp.columns = ['freq', 'type', 'ST','Patient', 'PSD_mean','PSD_median']


    
    return tp


def get_foldernames(x):

    first, last = os.path.split(x.filename)

    name = first.split('/')[-2]
    
    return name




if __name__ == '__main__':
    
    main()



        


    