#!/usr/bin/env python3

if __name__ == '__main__':
    import os
    import sys
    import time
    import pandas as pd

    start_time = time.time()
    
    try:
        data_dir = sys.argv[1]
        out_dir = sys.argv[2]
    except:
        print('Usage: one_csv <dataset_dir> <out_dir>')
    else:
        if os.path.exists(data_dir):
            os.makedirs(out_dir, exist_ok=True)
            
            # list to save all dataframes generated from the csvs
            dfs = []
            
            for thisdir, subdirs, files in os.walk(data_dir):
                os.chdir(thisdir)
                for file in files:
                    if file.lower().endswith('.csv'):
                        print('Processing', file)
                        
                        # read and append
                        columns_names = [f"f{i}" for i in range(0, 554)]  # first column gets deleted that's why I start at 1
                        df = pd.read_csv(file, header=None, names=columns_names)
                        df = df.drop(df.columns[0], axis=1) # drop first column --> contains index numbers
                        dfs.append(df)
                        
                        print('Done!\n')
            
            # combine all dfs into one
            combined_df = pd.concat(dfs, ignore_index=True)
            #print(combined_df) # print test
            
            # saving file
            panda_file_name = os.path.join(out_dir, "data.csv")
            combined_df.to_csv(panda_file_name, columns=columns_names[1:], header=True, index=False)
            
            end_time = time.time()

            print(f"File generated! Script ran in {end_time-start_time} seconds")
        else:
            sys.exit('Invalid Path!')
