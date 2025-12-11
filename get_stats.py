import pandas as pd
from scipy import stats
import compare_bbm_distribution as c
import sys

# Redirect stdout to null to suppress loading logs
import os
sys.stdout = open(os.devnull, 'w')

try:
    df_baru = c.load_data('Data W9371UM ULTRASONIK.xlsx', 'W9371UM BARU')
    df_lama = c.load_data('Data W9371UM ULTRASONIK.xlsx', 'W9371UM LAMA')
    trips_baru = c.calculate_trips(df_baru)
    trips_lama = c.calculate_trips(df_lama)
    
    u_stat, u_p = stats.mannwhitneyu(trips_baru['consumption_kmL'], trips_lama['consumption_kmL'], alternative='two-sided')
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    with open('final_stats.txt', 'w') as f:
        f.write(f"MW_U_STAT: {u_stat}\n")
        f.write(f"MW_P_VALUE: {u_p}\n")
        f.write(f"NEW_MEAN: {trips_baru['consumption_kmL'].mean()}\n")
        f.write(f"OLD_MEAN: {trips_lama['consumption_kmL'].mean()}\n")

except Exception as e:
    sys.stdout = sys.__stdout__
    print(e)
