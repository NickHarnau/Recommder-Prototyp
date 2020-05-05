import operator

def normalized_df(user_interaction, fillna = True):

    means = user_interaction.mean(axis=1, numeric_only=True).to_numpy() # calculate the mean, ignore nan values
    normalized = user_interaction.sub(means, axis=0) # normalize the data by subtracting the value by the mean -> 0 becomes the mean for each user
    if fillna:
        normalized.fillna(0,inplace=True)# cause 0 is the mean - 0 do not manipulate the result anymoren ToDo: fillna richtig?
    return normalized, means



def recommender_items_CF(user_interaction_df, cosine_similarity, userID, similar_users=5):
    normal_with_cs = user_interaction_df.copy()
    normal_with_cs["similarity"] = cosine_similarity[user_interaction_df.index.get_loc(userID)]  # add cosine similarity of specific User to other Userts to df
    specific_user_interaction = user_interaction_df.loc[[userID]]  # get the relevant user
    not_interacted_items = specific_user_interaction.columns[specific_user_interaction.isnull().any()].tolist()  # get not interacted items of user
    """
    for each user choose n similar user (by the cosine similarity)
    then predict the score for the not interacted item by the user with the ratings of the similar users for this item
    """
    Recommender = {}
    for item in not_interacted_items:
        recommend_specific_item = normal_with_cs.dropna(subset=[item]).copy()
        s_u = recommend_specific_item.nlargest(similar_users + 1, ["similarity"])  # choose n similar user / +1, because traget user is included
        s_u = s_u.loc[s_u["similarity"] > 0]
        # get for each not interacted Item a recommendation score
        product_cos_item = sum(s_u[item] * s_u["similarity"])
        """because of normalization negative cosine similarity is possible,
        because of that recommended scores out of the actual scoring range are possible
        to avoid that limit the recommendation score to the max (in this case 10) / min (in this case 1)"""
        if product_cos_item / s_u["similarity"].sum() > 10: # limit score to max score
            Recommender[item] = 10
            user_interaction_df.loc[userID, item] = 10
        elif product_cos_item / s_u["similarity"].sum() < 1: # limit score to min score
            Recommender[item] = 1
            user_interaction_df.loc[userID, item] = 1
        else:
            Recommender[item] = product_cos_item / s_u["similarity"].sum()
            user_interaction_df.loc[userID, item] = product_cos_item / s_u["similarity"].sum()
    # return a sorted dict and a DF with the predicted values filled in.
    return(dict(sorted(Recommender.items() ,key= operator.itemgetter(1), reverse=True))) , user_interaction_df

"""
# earlier approach for calculating Cosine Similarity for individual user.
Is now replaced by calculating the cosine similarity only once for all users and then just take from that the relevant user cosine similarity


def calculat_cosine_similarity(normalized_df, userID): ##

    X = normalized_df.loc[[userID]]
    Y = normalized_df.drop([userID])
    cs = cosine_similarity(X,Y)
    cs_with_object = np.insert(cs,userID,1) # 1 at the place of the user ID -> because cosine similarity with itself is 1 -> to keep the rigth order
    return cs_with_object

"""
