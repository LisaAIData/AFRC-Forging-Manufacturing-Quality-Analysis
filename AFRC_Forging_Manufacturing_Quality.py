#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import clustermap
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[32]:


cmm = pd.read_excel('CMMData.xlsx')
cmm


# In[33]:


# check cmm datasets
cmm.info()


# In[34]:


# check null values
cmm.isnull().sum() 
# from table below, all values are 0. We can see that there are no NA values in cmm datasets


# In[35]:


# rename features 
cmm.rename(columns={'Unnamed: 3':'Measurement'}, inplace=True)
cmm.drop(columns=['Unnamed: 0'], inplace=True)
cmm.drop(columns=['Unnamed: 1'], inplace=True)
cmm.drop(columns=['Unnamed: 2'], inplace=True)
cmm.iloc[0,0] = 'Nominal value'
cmm.iloc[1,0] = 'Upper error'
cmm.iloc[2,0] = 'Lower error'
cmm


# As each measurement has different standard, I will transfer the norminal value to the same standard which is 1. Therefore, each measurement values are around 1 with different variance.

# #### for loop, calculate the norminal values. For each column, I transfer all norminal value to 1 in order to getthe same norminal values. To do that, I divide all values with their corresponding column's norminal values. Bydoing so, all measurement will have the same standard value as 1, and each measurement value differs around 1.
# 

# In[36]:


# For loop, standardize norminal values into 1.
for i in range(1, len(cmm.columns)):
    cmm.iloc[:, i] = cmm.iloc[:, i] / cmm.iloc[0,i]
cmm


# In[37]:


# Build measurement standard data
cmm_measure = cmm.iloc[:3]
cmm_measure


# In[38]:


# Build measure values for each part ID
cmm_part = cmm.iloc[3:].rename(columns={'Measurement':'Part_ID'}) 
# Set ID index
cmm_part.set_index('Part_ID', inplace=True)
cmm_part


# #### To calculate the total error for each part ID. As errors in 18 measures differs in positive and negative values, simply sum the error will cause positive error to cancel out the negative ones. In addition, each measurements' error difference is very small. Thus, I will square each error in 18 measurements to magnify errors, and then take the sum as the total error for each part ID.

# In[39]:


cmm_part['total_error'] = 0
for i in range(0,len(cmm_part.columns)-1):
    # calculate the square for each error, minus norminal values 1
    cmm_part.iloc[:,i] = np.square(cmm_part.iloc[:,i]-1) 
    cmm_part['total_error'] = cmm_part.iloc[:,i] + cmm_part['total_error'] # calculate the sum of squared errors
    
cmm_part


# #### From the table above, we can see the squared error for each part and the total error for each Part ID.

# In[40]:


# Rank total errors
cmm_part.sort_values(by='total_error', inplace=True, ascending=False)
cmm_part[['total_error']].reset_index()


# #### From the table above, we can see that part ID 23, 28, 32  have the top three largest total errors. For these five part ID, they may have highest probability of bad quality. To further analyse which sensor parameters most affect these three parts, I will calculate clustermap to visualize the influence of parameters and total_error.

# In[41]:


# Read five bad quality parts
part23 = pd.read_csv('Scope0023.csv', encoding='unicode_escape')
part28 = pd.read_csv('Scope0028.csv', encoding='unicode_escape')
part32 = pd.read_csv('Scope0032.csv', encoding='unicode_escape')


# For sensor machine parameters, there are 99 parameters in total. However, not all parameters' status is 'in use'. I assume that these 'not in use' and 'unknow parameters' are old fasioned sensors which are not used. In this case, I will only analyse quality based on sensor parameters which is 'in use' status.

# In[42]:


# Check the parameter sheet
data = pd.ExcelFile('ForgedPartDataStructureSummaryv3.xlsx')
data_names = data.sheet_names
data_names


# In[43]:


# Only get 'Machine Parameters' dataset
for par in data_names:
    if par == 'Machine Parameters':
        parameter = pd.read_excel('ForgedPartDataStructureSummaryv3.xlsx', sheet_name=par)
parameter


# In[44]:


# Check columns of parameter datasets
parameter.columns


# In[45]:


# There are many sensor parameter status, I only choose parameter with 'In Use'
parameter['Classification'].unique()


# In[60]:


# Find signal name which classification is 'in use'
signal_name = parameter[parameter['Classification'] == 'In Use'] # Select all 'In Use' parameters
signal_name = signal_name[['Signal Name']] # only choose Signal Name column
signal_name


# In[47]:


# get signal_name parameters
signal_name['Signal Name'].unique()


# ### Drop signal names which parameters are not 'In Use'

# ### Part23

# In[48]:


# Select datasets which only 'In Use' signal name parameters
part23 = part23.loc[:,['Timer Tick [ms]', 'Power [kW]', 'Force [kN]', 'A_ges_vibr',
       'Schlagzahl [1/min]', 'EXZ_pos [deg]', 'A_ACTpos [mm]',
       'DB_ACTpos [mm]', 'L_ACTpos [mm]', 'R_ACTpos [mm]',
       'SBA_ActPos [mm]', 'INDA_ACTpos [deg]', 'A_ACT_Force [kN]',
       'DB_ACT_Force [kN]', 'L_ACTspd [mm/min]', 'R_ACTspd [mm/min]',
       'SBA_NomPos [mm] [mm]', 'A_ACTspd [mm/min]', 'DB_ACTspd [mm/min]',
       'L_NOMpos [mm]', 'R_NOMpos [mm]', 'SBA_OUT [%]', 'A_NOMpos [mm]',
       'DB_NOMpos [mm]', 'L_OUT [%]', 'R_OUT [%]', 'Feedback SBA [%]',
       'A_OUT [%]', 'DB_OUT [%]', 'L_NOMspd [mm/min]',
       'R_NOMspd [mm/min]', 'Frc_Volt', 'A_NOMspd [mm/min]',
       'DB_NOMspd [mm/min]', 'Feedback L [%]', 'Feedback R [%]',
       'Speed Vn_1 [rpm]', 'NOMforceSPA [kN]', 'IP_ActSpd [mm/min]',
       'IP_ActPos [mm]', 'SPA_OUT [%]', 'Feedback A [%]',
       'Feedback DB [%]', 'IP_NomSpd [mm/min]', 'IP_NomPos',
       'Feedback_SPA [%]', 'ForgingBox_Temp', 'TMP_Ind_U1 [°C]',
       'TMP_Ind_F [°C]', 'IP_Out [%]', 'ACTforceSPA [kN]',
       '$U_GH_HEATON_1 (U25S0).1']]
part23


# In[49]:


# As each value differ, I use normalization to rerange all values into [0,1] field
transfer = MinMaxScaler()
part23_new = transfer.fit_transform(part23)      # Normalize part23 values
part23_new = pd.DataFrame(part23_new, columns = part23.columns)   # Transfer to DataFrame
part23_error = cmm_part['total_error'].iloc[0]   # Get the Part23's total error
part23_new['part23_error'] = part23_error        # Create a new column for part 23's error
part23_new.head()



x,y= part23_new.iloc[:, 0:-1], part23_new.loc[:, 'part23_error'] # get sensor values as x,  part23_error as y
sensors = list(part23_new.iloc[:, 0:-1].columns) # get feature names


# In[68]:


# get correlation analysis
plt.figure(figsize = (40,30))
colmn = part23_new.columns.tolist() # list name
mcorr = part23_new[colmn].corr(method='spearman') # get the spearman correlation axis

mask = np.zeros_like(mcorr, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True #Triangle size
cmap = sns.diverging_palette(220, 10, as_cmap=True) # return matplotlib colormap figures
g = sns.heatmap(mcorr,mask = mask, cmap = cmap, square=True, annot=True, fmt='0.2f') #get the heatmap of correlation
plt.show()


# In[50]:


x,y= part23_new.iloc[:, 0:-1], part23_new.loc[:, 'part23_error']


# In[51]:


sns.clustermap(data=part23_new, pivot_kws=None, method='average', 
              figsize=(5, 5), cbar_kws=None, 
               row_cluster=False, col_cluster=True)


# In[ ]:





# ### Part28

# In[52]:


# Select datasets which only 'In Use' signal name parameters
part28 = part28.loc[:,['Timer Tick [ms]', 'Power [kW]', 'Force [kN]', 'A_ges_vibr',
       'Schlagzahl [1/min]', 'EXZ_pos [deg]', 'A_ACTpos [mm]',
       'DB_ACTpos [mm]', 'L_ACTpos [mm]', 'R_ACTpos [mm]',
       'SBA_ActPos [mm]', 'INDA_ACTpos [deg]', 'A_ACT_Force [kN]',
       'DB_ACT_Force [kN]', 'L_ACTspd [mm/min]', 'R_ACTspd [mm/min]',
       'SBA_NomPos [mm] [mm]', 'A_ACTspd [mm/min]', 'DB_ACTspd [mm/min]',
       'L_NOMpos [mm]', 'R_NOMpos [mm]', 'SBA_OUT [%]', 'A_NOMpos [mm]',
       'DB_NOMpos [mm]', 'L_OUT [%]', 'R_OUT [%]', 'Feedback SBA [%]',
       'A_OUT [%]', 'DB_OUT [%]', 'L_NOMspd [mm/min]',
       'R_NOMspd [mm/min]', 'Frc_Volt', 'A_NOMspd [mm/min]',
       'DB_NOMspd [mm/min]', 'Feedback L [%]', 'Feedback R [%]',
       'Speed Vn_1 [rpm]', 'NOMforceSPA [kN]', 'IP_ActSpd [mm/min]',
       'IP_ActPos [mm]', 'SPA_OUT [%]', 'Feedback A [%]',
       'Feedback DB [%]', 'IP_NomSpd [mm/min]', 'IP_NomPos',
       'Feedback_SPA [%]', 'ForgingBox_Temp', 'TMP_Ind_U1 [°C]',
       'TMP_Ind_F [°C]', 'IP_Out [%]', 'ACTforceSPA [kN]',
       '$U_GH_HEATON_1 (U25S0).1']]
part28


# In[53]:


# As each value differ, I use normalization to rerange all values into [0,1] field
transfer = MinMaxScaler()
part28_new = transfer.fit_transform(part28)      # Normalize part28 values
part28_new = pd.DataFrame(part28_new, columns = part23.columns)            # Transfer to DataFrame
part28_error = cmm_part['total_error'].iloc[1]   # Get the Part28's total error, which is the second row
part28_new['part28_error'] = part28_error        # Create a new column for part 28's error
part28_new.head()




# get correlation analysis
plt.figure(figsize = (40,30))
colmn = part28_new.columns.tolist() # list name
mcorr = part28_new[colmn].corr(method='spearman') # get the spearman correlation axis

mask = np.zeros_like(mcorr, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True #Triangle size
cmap = sns.diverging_palette(220, 10, as_cmap=True) # return matplotlib colormap figures
g = sns.heatmap(mcorr,mask = mask, cmap = cmap, square=True, annot=True, fmt='0.2f') #get the heatmap of correlation
plt.show()

# In[54]:


x,y= part28_new.iloc[:, 0:-1], part28_new.loc[:, 'part28_error']


# In[55]:


sns.clustermap(data=part28_new, pivot_kws=None, method='average', 
              figsize=(5, 5), cbar_kws=None, 
               row_cluster=False, col_cluster=True)


# In[ ]:





# In[ ]:





# ### Part32

# In[56]:


# Select datasets which only 'In Use' signal name parameters
part32 = part32.loc[:,['Timer Tick [ms]', 'Power [kW]', 'Force [kN]', 'A_ges_vibr',
       'Schlagzahl [1/min]', 'EXZ_pos [deg]', 'A_ACTpos [mm]',
       'DB_ACTpos [mm]', 'L_ACTpos [mm]', 'R_ACTpos [mm]',
       'SBA_ActPos [mm]', 'INDA_ACTpos [deg]', 'A_ACT_Force [kN]',
       'DB_ACT_Force [kN]', 'L_ACTspd [mm/min]', 'R_ACTspd [mm/min]',
       'SBA_NomPos [mm] [mm]', 'A_ACTspd [mm/min]', 'DB_ACTspd [mm/min]',
       'L_NOMpos [mm]', 'R_NOMpos [mm]', 'SBA_OUT [%]', 'A_NOMpos [mm]',
       'DB_NOMpos [mm]', 'L_OUT [%]', 'R_OUT [%]', 'Feedback SBA [%]',
       'A_OUT [%]', 'DB_OUT [%]', 'L_NOMspd [mm/min]',
       'R_NOMspd [mm/min]', 'Frc_Volt', 'A_NOMspd [mm/min]',
       'DB_NOMspd [mm/min]', 'Feedback L [%]', 'Feedback R [%]',
       'Speed Vn_1 [rpm]', 'NOMforceSPA [kN]', 'IP_ActSpd [mm/min]',
       'IP_ActPos [mm]', 'SPA_OUT [%]', 'Feedback A [%]',
       'Feedback DB [%]', 'IP_NomSpd [mm/min]', 'IP_NomPos',
       'Feedback_SPA [%]', 'ForgingBox_Temp', 'TMP_Ind_U1 [°C]',
       'TMP_Ind_F [°C]', 'IP_Out [%]', 'ACTforceSPA [kN]',
       '$U_GH_HEATON_1 (U25S0).1']]
part32


# In[57]:


# As each value differ, I use normalization to rerange all values into [0,1] field
transfer = MinMaxScaler()
part32_new = transfer.fit_transform(part32)      # Normalize part32 values
part32_new = pd.DataFrame(part32_new, columns = part32.columns)            # Transfer to DataFrame
part32_error = cmm_part['total_error'].iloc[2]   # Get the Part32's total error, which is the third row
part32_new['part32_error'] = part32_error        # Create a new column for part 32's error
part32_new.head()


# In[58]:

# get correlation analysis
plt.figure(figsize = (40,30))
colmn = part32_new.columns.tolist() # list name
mcorr = part32_new[colmn].corr(method='spearman') # get the spearman correlation axis

mask = np.zeros_like(mcorr, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True #Triangle size
cmap = sns.diverging_palette(220, 10, as_cmap=True) # return matplotlib colormap figures
g = sns.heatmap(mcorr,mask = mask, cmap = cmap, square=True, annot=True, fmt='0.2f') #get the heatmap of correlation
plt.show()

x,y= part32_new.iloc[:, 0:-1], part32_new.loc[:, 'part32_error']


# In[59]:


sns.clustermap(data=part32_new, pivot_kws=None, 
              figsize=(5, 5), cbar_kws=None, 
               row_cluster=False, col_cluster=True)


# In[ ]:





# In[ ]:





# In[ ]:




