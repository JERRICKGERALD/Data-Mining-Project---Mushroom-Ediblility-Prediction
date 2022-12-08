# EDA for does-bruise-or-bleed, season, habitat,  stem_width 
# By: Lijia Ren

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py

#%%
df = pd.read_csv('final.csv')
df.info()
print(df.head())


#%% [markdown] 
# Does-Bruise-Or-Bleed Distribution

#%%
sns.displot(df, x="does-bruise-or-bleed", hue="class", multiple="stack", palette="husl").set(title="Does-Bruise-Bleed Distribution")

#%%
table2 = pd.crosstab(df['does-bruise-or-bleed'], 
                            df['class'],
                                margins = False)

table2['total'] = table2.sum(axis=1)
table2['% e'] = (table2['e'] / table2.total * 100).round(2)
table2['% p'] = (table2['p'] / table2.total * 100).round(2)
table2.sort_values('total', ascending=False)

#%% [markdown] 
# #### Does-Bruise-Or-Bleed Observations
# - The majority of mushroom Does-Bruise-Or-Bleed are no Bruis is no Bruise-Or-Bleed (f); however, 
# there is no big difference between " no Bruise-Or-Bleed (f) " and " Bruise-Or-Bleed (t)".


#%% [markdown] 
# Season Distribution

#%%
sns.displot(df, x="season", hue="class", multiple="stack", palette="husl").set(title="Season Distribution")

#%%
table2 = pd.crosstab(df['season'], 
                            df['class'],
                                margins = False)

table2['total'] = table2.sum(axis=1)
table2['% e'] = (table2['e'] / table2.total * 100).round(2)
table2['% p'] = (table2['p'] / table2.total * 100).round(2)
table2.sort_values('total', ascending=False)

#%% [markdown] 
# ####Season Observations
# - The majority of season is autumn (a)
# - There is no season with a high proportion (>70%) of poisonous mushrooms.


#%% [markdown] 
# Habitat Distribution

#%%
sns.displot(df, x="habitat", hue="class", multiple="stack", palette="husl").set(title="Habitat Distribution")

#%%
table2 = pd.crosstab(df['habitat'], 
                            df['class'],
                                margins = False)

table2['total'] = table2.sum(axis=1)
table2['% e'] = (table2['e'] / table2.total * 100).round(2)
table2['% p'] = (table2['p'] / table2.total * 100).round(2)
table2.sort_values('total', ascending=False)

#%% [markdown] 
# ####Habitat Observations
# - The majority of habitat is  (d)
# - The habitat with a high proportion (>70%) of poisonous mushrooms is (p)
# - The habitat with a high proportion (>70%) of edible mushrooms is (w) and (u)


#%% [markdown] 
# Stem-Width Box-Plot


#%%



sns.boxplot(y='stem-width',x='class',data=df)
#%% [markdown] 
# #### Steam_Width  Observations
# - The average stem-width with a high proportion of poisonous mushrooms is shorter than the edible mushrooms.