#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from scipy.integrate import trapezoid
from scipy.integrate import simps
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Path to your Excel file
file_path = '/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/Ex_12.xlsx'

# Read the Excel file, specifying a sheet and columns
df = pd.read_excel(file_path, sheet_name='Sheet1')
df.head()


# In[ ]:


df.shape


# In[ ]:


df1=df.copy()
df1.head()


# In[ ]:


df1['datetime'] = pd.to_datetime(df1['date'].astype(str) + ' ' + df1['time'].astype(str))

# Sort the DataFrame by datetime
df2 = df1.sort_values(by='datetime')
df2.head()


# In[ ]:


first_timestamp = df2['datetime'].iloc[0]
df2['timedelta_seconds'] = (df2['datetime'] - first_timestamp).dt.total_seconds()
print(df2.timedelta_seconds.unique())


# In[ ]:


df2.head()


# In[ ]:


energy_in_mWs = trapezoid(df2.current_power,df2.timedelta_seconds)
energy_in_Wh = energy_in_mWs/(1e3*3600) # mWs -> Wh
print(energy_in_Wh)


# In[ ]:


rough_energy_estimation=df2.current_power.sum()/(1e3*3600)
print(f'The print took around {round(energy_in_Wh,3)}Wh of energy, or {round(energy_in_Wh*3600,3)}J in {int(df2.timedelta_seconds.iloc[-1])} seconds')


# In[ ]:


file_path = '/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/Ex_13.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
df1=df.copy()
df1['datetime'] = pd.to_datetime(df1['date'].astype(str) + ' ' + df1['time'].astype(str))
df2 = df1.sort_values(by='datetime')
first_timestamp = df2['datetime'].iloc[0]
df2['timedelta_seconds'] = (df2['datetime'] - first_timestamp).dt.total_seconds()
energy_in_mWs = trapezoid(df2.current_power,df2.timedelta_seconds)
energy_in_Wh = energy_in_mWs/(1e3*3600) # mWs -> Wh
rough_energy_estimation=df2.current_power.sum()/(1e3*3600)
print(f'The print took around {round(energy_in_Wh,3)}Wh of energy, or {round(energy_in_Wh*3600,3)}J in {int(df2.timedelta_seconds.iloc[-1])} seconds')


# In[ ]:





# In[ ]:





# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/Ex_14.csv')
df.head()


# In[ ]:


# # Define the directory containing the CSV files
# directory = '/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/'

# # List all files in the directory
# files = os.listdir(directory)

# # Filter out only the CSV files
# csv_files = [file for file in files if file.endswith('.csv')]

# # Read the head of each CSV file and store it in a dictionary
# dataframes = {}
# for csv_file in csv_files:
#     file_path = os.path.join(directory, csv_file)
#     df = pd.read_csv(file_path)
#     dataframes[csv_file] = df.head()

# # Print the head of each DataFrame
# for csv_file, df_head in dataframes.items():
#     print(f"Head of {csv_file}:")
#     print(df_head)
#     print("\n")


# In[ ]:


# Define the directory containing the CSV files
directory = '/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/'

# List all files in the directory
files = os.listdir(directory)

# Filter out only the CSV files that start with 'Ex_'
csv_files = [file for file in files if file.startswith('Ex_') and file.endswith('.csv')]

# Read the head of each CSV file and store it in a dictionary
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    df = pd.read_csv(file_path)
    df = df.drop(columns=['month_energy', 'month_runtime', 'today_energy', 'today_runtime'])

    print(f"Head of {csv_file}:")
    print(df.head())
    print("\n")


# In[ ]:


file_path='/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/real_time_data_3D_printing.xlsx'
df_3d_printer = pd.read_excel(file_path, sheet_name='Sheet1')
df_3d_printer.head()


# In[ ]:


df_3d_printer.columns


# In[ ]:


df_3d_printer=df_3d_printer.drop(columns=['Estimated time', 'Wt-g', 'L-m', 'End Time (hh:mm)',
       'Print Time (mm:ss)', 'Layer Height', 'Small Hole Max Size',
       'Outer Wall Speed', 'Inner Wall Speed', 'Travel Speed',
       'Outer Wall Line Width', 'Print Speed', 'Outer Wall Inset',
       'Initial Layer Speed', 'Wall Line Width', 'Make Overhang Printable',
       'Use Adaptive Layers', 'Wipe Nozzle Between Layers',
       'Alternate Wall Directions', 'Group Outer Walls', 'Slicing Tolerance',
       'Generate Support', 'Build Plate Adhesion'])
df_3d_printer.head()


# In[ ]:


df_3d_printer['Experiment No'] = df_3d_printer['Experiment No'].apply(lambda x: f'Ex_{x}.csv')
df_3d_printer.head()


# **Formulas**
# 
# Convert watt-hours to kilowatt-hours:
# Energy in kWh= Energy in Wh / 1000
# 
# Convert joules to kilowatt-hours:
# Energy in kWh=Energy in J/(3.6×10^6)
# 
# Calculate CO2 emissions:
# CO2 emissions (g)=Energy in kWh x Carbon Intensity (gCO2/kWh)
# 
# Carbon Intensity (gCO2/kWh) = 300

# Finding area of the graph of energy consumption to get the overall energy consumed for the given experiment using trapezoid integration function.

# In[ ]:


# Define the directory containing the CSV files
directory = '/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/'

# List all files in the directory
files = os.listdir(directory)

# Filter out only the CSV files that start with 'Ex_'
csv_files = [file for file in files if file.startswith('Ex_') and file.endswith('.csv')]

# Read the head of each CSV file and store it in a dictionary
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    df = pd.read_csv(file_path)
    df = df.drop(columns=['month_energy', 'month_runtime', 'today_energy', 'today_runtime'])
    df['local_time'] = pd.to_datetime(df['local_time'])
    df['date'] = df['local_time'].dt.date
    df['time'] = df['local_time'].dt.time
    df = df.drop(columns=['local_time'])

    # print(f"Head of {csv_file}:")
    # print(df.head())
    matched_row = df_3d_printer[df_3d_printer['Experiment No'] == csv_file]
    print_start_time = matched_row['Printing Start Time (hh:mm:ss)'].iloc[0]
    # print(f"{csv_file}:")
    # print(print_start_time)
    df_matched_row = df[df['time'] == print_start_time]
    # print(df_matched_row)
    # print(df_matched_row.iloc[0].name)
    # print(type(df_matched_row.iloc[0].name))
    # print(df.shape)
    df_new=df.iloc[df_matched_row.iloc[0].name:]
    # print(df_new.shape)
    # print(df_new.head())
    df_new = df_new.reset_index(drop=True)
    # print(df_new.shape)
    # print(df_new.head())
    df_new['datetime'] = pd.to_datetime(df_new['date'].astype(str) + ' ' + df_new['time'].astype(str))
    # Sort the DataFrame by datetime
    df_new = df_new.sort_values(by='datetime')
    # print(df_new.head())
    # print(df_new.isnull().sum())

    # Calculate the timedelta_seconds column
    first_timestamp = df_new['datetime'].iloc[0]
    df_new['timedelta_seconds'] = (df_new['datetime'] - first_timestamp).dt.total_seconds()
    # print(df_new.timedelta_seconds.unique())
    energy_in_mWs = trapezoid(df_new.current_power,df_new.timedelta_seconds)
    energy_in_Wh = energy_in_mWs/(1e3*3600) # mWs -> Wh
    rough_energy_estimation=df_new.current_power.sum()/(1e3*3600)
    print(f'{csv_file}:The print took around {round(energy_in_Wh,3)}Wh of energy, or {round(energy_in_Wh*3600,3)}J in {int(df_new.timedelta_seconds.iloc[-1])} seconds')
    # print(df_new['current_power'].describe())

    # Plot the energy consumption
    plt.figure(figsize=(12, 6))
    plt.plot(df_new['datetime'], df_new['current_power'], label='Current Power')
    plt.xlabel('Time')
    plt.ylabel('Current Power (mW)')
    plt.title(f'Energy Consumption Over Time - {csv_file}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("\n")


# Finding area of the graph of energy consumption to get the overall energy consumed for the given experiment using simps integration function.

# In[ ]:


# Define the directory containing the CSV files
directory = '/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/power_consumption_data/'

# List all files in the directory
files = os.listdir(directory)

# Filter out only the CSV files that start with 'Ex_'
csv_files = [file for file in files if file.startswith('Ex_') and file.endswith('.csv')]

# Read the head of each CSV file and store it in a dictionary
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    df = pd.read_csv(file_path)
    df = df.drop(columns=['month_energy', 'month_runtime', 'today_energy', 'today_runtime'])
    df['local_time'] = pd.to_datetime(df['local_time'])
    df['date'] = df['local_time'].dt.date
    df['time'] = df['local_time'].dt.time
    df = df.drop(columns=['local_time'])

    # print(f"Head of {csv_file}:")
    # print(df.head())
    matched_row = df_3d_printer[df_3d_printer['Experiment No'] == csv_file]
    print_start_time = matched_row['Printing Start Time (hh:mm:ss)'].iloc[0]
    print(f"{csv_file}:")
    # print(print_start_time)
    df_matched_row = df[df['time'] == print_start_time]
    # print(df_matched_row)
    # print(df_matched_row.iloc[0].name)
    # print(type(df_matched_row.iloc[0].name))
    # print(df.shape)
    df_new=df.iloc[df_matched_row.iloc[0].name:]
    # print(df_new.shape)
    # print(df_new.head())
    df_new = df_new.reset_index(drop=True)
    # print(df_new.shape)
    # print(df_new.head())
    df_new['datetime'] = pd.to_datetime(df_new['date'].astype(str) + ' ' + df_new['time'].astype(str))
    # Sort the DataFrame by datetime
    df_new = df_new.sort_values(by='datetime')
    # print(df_new.head())
    # print(df_new.isnull().sum())

    # Calculate the timedelta_seconds column
    first_timestamp = df_new['datetime'].iloc[0]
    df_new['timedelta_seconds'] = (df_new['datetime'] - first_timestamp).dt.total_seconds()
    # print(df_new.timedelta_seconds.unique())
    energy_in_mWs = simps(df_new['current_power'], df_new['timedelta_seconds'])
    energy_in_Wh = energy_in_mWs/(1e3*3600) # mWs -> Wh
    rough_energy_estimation=df_new.current_power.sum()/(1e3*3600)
    print(f'The print took around {round(energy_in_Wh,3)}Wh of energy, or {round(energy_in_Wh*3600,3)}J in {int(df_new.timedelta_seconds.iloc[-1])} seconds')
    # print(df_new['current_power'].describe())

    # Plot the energy consumption
    plt.figure(figsize=(12, 6))
    plt.plot(df_new['datetime'], df_new['current_power'], label='Current Power')
    plt.xlabel('Time')
    plt.ylabel('Current Power (mW)')
    plt.title(f'Energy Consumption Over Time - {csv_file}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




