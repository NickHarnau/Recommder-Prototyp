import numpy as np
# create a df/matrix with features per Item
def content_based_user(userID, user_interaction_df, feature_df): # with full user_interaction_df - can contain nan
    """
    user_interaction = DF with User in row and Items in column. Value is the score, a user gives a item
    feature_df = DF with Items in row and feature in columns. Value is 1 if a item has a feature

    Calculates for a specific User the predicted rating of not interacted Ratings, depending of the history of the user.
    Each Item is described by features. So each user_interaction with an item, shows how the user likes the corresponding features.

    A weighted score is calculated for all features. The higher the score, the more important the feature is for the user.
    The prediction for an item is the sum of the weightes score of all features, that fits to an item.

    """
    specific_user_interaction = user_interaction_df.loc[[userID]] # get the relevant user
    not_interacted_items = specific_user_interaction.columns[specific_user_interaction.isnull().any()].tolist() # items with whom has not yet interacted with as list
    relevant_user_interacted = specific_user_interaction.drop(not_interacted_items, axis=1) #only keep interacted items in specific user/interaction df
    relevant_user_array = relevant_user_interacted.to_numpy()# get interacted items of user as array
    interacted_items = feature_df.drop(not_interacted_items, axis=0).to_numpy()# drop the items of item feature with whom the user has not interacted with and transform to array

    weighted_features = calculate_weighted_features(relevant_user_array, interacted_items)

    df_item_notwatched = feature_df.loc[not_interacted_items]
    recommendation = df_item_notwatched * weighted_features * 10 ## *10 to get the predicted ratings in range of the possible ratings
    recommendation_score = recommendation.sum(axis=1).sort_values(ascending=False)# sort after recommendation for the item
    for value in range(len(recommendation_score)):
        column = recommendation_score.index[value]
        user_interaction_df.loc[userID,column] = recommendation_score[value]
    return recommendation_score.to_dict(), user_interaction_df

def calculate_weighted_features(user_array, interacted_item_array): # calculates the weighted importance of the features
    relevant_features = np.matmul(user_array, interacted_item_array)
    weighted_features = np.true_divide(relevant_features, np.sum(relevant_features))
    return weighted_features





