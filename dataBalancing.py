import pandas as pd
from pandas import read_csv
import numpy as np
df = read_csv('dataNormalized.csv')

balanced = pd.DataFrame() 	# Balanced dataset dataframe
bins = 1000 				# 1000 of bins
bin_n = 300 				# 300 frames in each bin (at most)

start = 0
for end in np.linspace(0, 1, num=bins):  
    df_range = df[(np.absolute(df.steering) >= start) & (np.absolute(df.steering) < end)]
    print(len(df_range))
    range_n = min(bin_n, df_range.shape[0])
    balanced = pd.concat([balanced, df_range.sample(range_n)])
    start = end
balanced.to_csv('driving_log_balanced.csv', index=False)