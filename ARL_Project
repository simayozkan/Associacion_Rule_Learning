#ARL
#The dataset is not shared because it's exclusive to Miuul Data Science Bootcamp

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
!pip install mlxtend
pip install openpyxl
import warnings
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("dataset")
df

df["Service"] = df['ServiceId'].astype(str) + "_" + df['CategoryId'].astype(str)

df['New_Date'] = pd.DatetimeIndex(df['CreateDate']).year.astype(str) + "_" + pd.DatetimeIndex(df['CreateDate']).month.astype(str)

df["SepetID"] = df['UserId'].astype(str) + "_" + df['New_Date'].astype(str)

basket_service_df = df.groupby(['SepetID', 'Service'])['Service'].count().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(sepet_hizmet_df,
                            min_support=0.01,
                            use_colnames=True)


rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

#recommendation
arl_recommender(rules, "2_0", 3)
