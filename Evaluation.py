from EvaluateFunction import *
from Main import *
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


train_set, sample = random_train_set(user_interaction, 0.8) # create random test set
Range = 1
"""
Evaluation Scores: 
MSE: the smaller the better
r2: the closer to 1 the better 
EVS: the closer to 1 the better
"""
CB_dict = {}
CF_dict = {}
MF_dict = {}
# CF MSE
train_set_CF = train_set.copy()
normalized_test , means = normalized_df(train_set, fillna = True) # normalize the test set anf get the means
for row in normalized_test.index.tolist():
    """
    For each row/index (in my Example = UserId) calculates the cosine similarity 
    and the calculate the predicted rating for not yet interacted items 
    add the predicted rating to df and get a full df
    """
    train_set_CF = recommender_items_CF(train_set_CF, cs, row, similar_users=3)[1]

predicted_values_CF , true_values = predicted_true_value(train_set_CF, user_interaction, sample)
# different evaluation approaches
MSE_CF = mean_squared_error(true_values, predicted_values_CF)
CF_dict["MSE"]=MSE_CF
r2_CF = r2_score(true_values,predicted_values_CF)
CF_dict["r2"]=r2_CF
EVS_CF = explained_variance_score(true_values, predicted_values_CF)
CF_dict["EVS"]=EVS_CF
count_correct_CF, procentual_correct_CF = count_close_predictions(true_values, predicted_values_CF, Range)
CF_dict["Correct_Count_rangeOf:{}".format(Range)]=count_correct_CF
CF_dict["Correct_Percent_rangeOf:{}".format(Range)]=procentual_correct_CF

# MF
"""
Matrix Factorization already provide us with predicted ratings 
"""
train_set_MF = matrix_factorization(normalized_test, 8, means, add_mean=True)
predicted_values_MF, true_values = predicted_true_value(train_set_MF, user_interaction, sample)
# differente evaluation approaches
MSE_MF = mean_squared_error(true_values, predicted_values_MF)
MF_dict["MSE"]=MSE_MF
r2_MF = r2_score(true_values,predicted_values_MF)
MF_dict["r2"]=r2_MF
EVS_MF = explained_variance_score(true_values, predicted_values_MF)
MF_dict["EVS"]=EVS_MF
count_correct_MF, procentual_correct_MF = count_close_predictions(true_values, predicted_values_MF, Range)
MF_dict["Correct_Count_rangeOf:{}".format(Range)]=count_correct_MF
MF_dict["Correct_Percent_rangeOf:{}".format(Range)]=procentual_correct_MF



# CB

train_cb = train_set.copy()
for row in train_cb.index.tolist():
    """
    similar to CF : Calculate the predicted ratings for each row/index(=userId) and fill them in df
    """
    train_cb = content_based_user(userID=row, user_interaction_df=train_cb, feature_df=feature_df)[1]

predicted_values_CB, true_values = predicted_true_value(train_cb, user_interaction, sample)
#different evaluation approaches
MSE_CB = mean_squared_error(true_values, predicted_values_CB)
CB_dict["MSE"]=MSE_CB
r2_CB = r2_score(true_values,predicted_values_CB)
CB_dict["r2"]=r2_CB
EVS_CB = explained_variance_score(true_values, predicted_values_CB)
CB_dict["EVS"]=EVS_CB
count_correct_CB, procentual_correct_CB = count_close_predictions(true_values, predicted_values_CB, Range)
CB_dict["Correct_Count_rangeOf:{}".format(Range)]=count_correct_CB
CB_dict["Correct_Percent_rangeOf:{}".format(Range)]=procentual_correct_CB

# store the different evaluation results in DF
evaluation_df = pd.DataFrame([CF_dict, CB_dict, MF_dict]).transpose()
evaluation_df.columns=["CF", "CB", "MF"]

print("Der Ansatz mit dem niedtigsten MSE ist: {} mit einem Wert von {}".format(evaluation_df.idxmin(axis=1)["MSE"],evaluation_df.min(axis=1)["MSE"]))
print("Der Ansatz mit dem höchsten r2 ist: {} mit einem Wert von {}".format(evaluation_df.idxmax(axis=1)["r2"],evaluation_df.max(axis=1)["r2"]))
print("Der Ansatz mit dem höchsten EVS ist: {} mit einem Wert von {}".format(evaluation_df.idxmax(axis=1)["EVS"],evaluation_df.max(axis=1)["EVS"]))
print("Der Ansatz mit dem höchsten richtig vorhergesagten Werte in einer Range von {} ist: {} mit einem Wert von {} \n"
      "Das entspricht einem Prozentwert von {}".format(Range, evaluation_df.idxmax(axis=1)["Correct_Count_rangeOf:1"],evaluation_df.max(axis=1)["Correct_Count_rangeOf:1"],
                                                       evaluation_df.max(axis=1)["Correct_Percent_rangeOf:1"]))

