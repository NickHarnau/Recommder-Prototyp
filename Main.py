import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Generate_DF import *
from ContentBased import *
from CollaborativeFiltering import *
from MatrixFactor import *




# 1. Base for all approach a random user/interaction df
user_interaction = rand_User_Interaction_df(rating_min=1, rating_max=10, rows=20, columns=20, nan_perc=0.3) # generate random user/interaction df
for row in user_interaction.index.tolist():
    if user_interaction.count(axis=1)[row]<10:
        user_interaction.drop(row)

# 2. Content Based Approach
feature_df = rand_feature_df (rows=20, columns=20) # generate random feature df / rows = columns in user_interaction
user_interaction_cb = user_interaction.copy()
recommendation_cb, user_interaction_cb = content_based_user(userID=2, user_interaction_df=user_interaction_cb, feature_df=feature_df) # get the recommendation as dict and the values directly filled in user/interaction df

# 3. Collaborative Filtering Approach
## cold start problem --> add fuction that only uses User with at least n interactions.
user_interaction_cf = user_interaction.copy()
normalized, means = normalized_df(user_interaction_cf, fillna = True) # normalize the user/interaction df and get the means / normalize only need to get the cs
cs = cosine_similarity(normalized) # calculates cosine similarity from all users to each other (include self) -
recommendation_cf , user_interaction_cf = recommender_items_CF(user_interaction_cf, cs, 2, similar_users=3) # get recommended Items

# 4. Matrix Factorisation approach

user_interaction_mf = matrix_factorization(normalized, 8, means, add_mean=True)# get the recommendation matrix with matrix factorisation
recommendation_mf = mf_recommendation(user_interaction_mf, user_interaction, 2) # get the best recommended Items as dict

