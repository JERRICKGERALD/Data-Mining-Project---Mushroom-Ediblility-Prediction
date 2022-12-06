#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("final.csv")
# %%
df.head()
# %%
# Stem height
plt.hist(df['stem-height'])

# The stem-height is approximately normally ditributed, with a slight right skew
# %%
# Stem width
plt.hist(df['stem-width'])

# the stem width is right skewed
# %%
# has_ring
counts = df['has-ring'].value_counts()
df.groupby(['has-ring']).size().plot(kind = "bar")
df.groupby(['has-ring', 'class']).size().plot(kind = "bar")
print(counts)

# there are 3 times are many mushrooms without a ring (f), so our data is unbalanced. However, they look fairly evenly split between the poisonous and edible mushrooms
# %%
# Ring type

df.groupby(['ring-type']).size().plot(kind = "bar")
counts = df['ring-type'].value_counts()
print(counts)

# the ring-type f is more common than all the other ring types combined. the next most common ring-types are e and z.
# %%
