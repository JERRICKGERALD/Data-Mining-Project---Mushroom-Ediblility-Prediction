# EDA for cap-shape, cap-color,gill-attachment, gill-color
# By: Karina Martinez

#%%
import numpy as np
import pandas as pd
import seaborn as sns

#%%
df = pd.read_csv('final.csv')
#df.info()

#%% [markdown] 
# #### Cap Shape Distribution

#%%
sns.displot(df, x="cap-shape", hue="class", multiple="stack", palette="husl").set(title="Cap Shape Distribution")

#%%
table1 = pd.crosstab(df['cap-shape'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%% [markdown] 
# #### Cap Shape Observations
# - The majority of mushroom cap shapes are convex (x) or flat (f)
# - The shapes with a high proportion (>70%) of poisonous mushrooms are bell (b) and others (o)
# - None of the shapes represent a high proportion of edible mushrooms

#%% [markdown] 
# #### Cap Color Distribution

#%%
sns.displot(df, x="cap-color", hue="class", multiple="stack", palette="husl").set(title="Cap Color Distribution")

#%%
table1 = pd.crosstab(df['cap-color'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%% [markdown] 
# #### Cap Color Observations
# - The majority of mushroom caps are brown (n), followed by yellow (y) and white (w)
# - The colors with a high proportion (>70%) of poisonous mushrooms are: red (e), orange (o), green (r), and pink (p)
# - The only color with a high proportion of edible mushrooms is buff (b)

#%% [markdown] 
# #### Gill Attachment Distribution

#%%
sns.displot(df, x="gill-attachment", hue="class", multiple="stack", palette="husl").set(title="Gill Attachment Distribution")

#%%

table1 = pd.crosstab(df['gill-attachment'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%% [markdown] 
# #### Gill Attachment Observations
# - The majority of gill attachments are adnate (a) or decurrent (d)
# - None of the gill attachments represent a high proportion of poisonous mushrooms
# - The only gill attachment with a high proportion (>70%) of edible mushrooms is pores (p)

#%% [markdown] 
# #### Gill Color Distribution

#%%
sns.displot(df, x="gill-color", hue="class", multiple="stack", palette="husl").set(title="Gill Color Distribution")

#%%

table1 = pd.crosstab(df['gill-color'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%% [markdown] 
# #### Gill Color Observations
# - The majority of gill colors is white (w)
# - The only gill color with a high proportion (>70%) of poisonous mushrooms is red (e)
# - The only gill color with a high proportion (>70%) of edible mushrooms is buff (b)

# %%