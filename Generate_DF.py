import numpy as np
import pandas as pd

def rand_User_Interaction_df(rating_min=1, rating_max=10, rows=10, columns=20, nan_perc=0.3): # max column = 24, cause columns = alphabet
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    content = np.random.randint(rating_min, rating_max, size=(rows, columns)).astype(float)
    mask = np.random.choice([1, 0], content.shape, p=[nan_perc, 1-nan_perc]).astype(bool)
    content[mask] = np.nan
    user_interaction = pd.DataFrame(content, columns=alphabet[:columns])
    return user_interaction


def rand_feature_df (rows=20, columns=20):## rows should be the same length as columns in rand_User_Interaction_df
    feature = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for letter in alphabet:
        feature.append(letter * 2)

    df_item_features = pd.DataFrame(np.random.randint(2, size=(rows, columns)), columns=feature[:rows])
    df_item_features = df_item_features.rename(index=dict(zip(range(rows), alphabet[:rows])))
    return df_item_features
