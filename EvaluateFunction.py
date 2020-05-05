import random
import numpy as np


def all_position_values(user_interaction_df):
    """
    get the position off all position containing a value (not na)
    returns a lists inside a list [[a,b],[c,d]...]
    On Place 0 if the nested list is the index(in my example = userID), on place 1 is the column
    """
    positions_with_values = []
    is_null = user_interaction_df.isnull()
    for column in is_null.columns:
        for row in is_null.index.tolist():
            if is_null.loc[row, column]==False:
                positions_with_values.append([row,column])
    return positions_with_values


def random_train_set(user_interaction_df, size_of_train_set): # replace at random positions the values with nan
    """
    Create a random train set
    Choose from the position with values randomly position and replace them by nan
    returns the train set and the position, which have been replaced
    """
    values_position = all_position_values(user_interaction_df)
    sample = random.choices(values_position, k=int((1-size_of_train_set)*len(values_position)))
    train_set = user_interaction_df.copy()
    for position in sample:
         train_set.loc[position[0],position[1]] = np.nan
    return train_set , sample # returns the with nan modified data set + the position of the modified position


def predicted_true_value(train_set, original_set,samples):
    """
    returns predicted and true values
    """
    predicted_values = []
    for position in samples:
        predicted_values.append(train_set.loc[position[0], position[1]])

    true_values = []
    for position in samples:
        true_values.append(original_set.loc[position[0], position[1]])

    return predicted_values,true_values

def count_close_predictions(true_value, predicted_value, range):
    """
    Count predicted values who are in a range of x around the true values
    """

    difference_absolut = np.absolute(np.array(true_value) - np.array(predicted_value))
    assumption = np.where(difference_absolut < range, 1, 0)
    Count_correct = (assumption == 1).sum()
    procentual_correct = Count_correct / len(difference_absolut)

    return Count_correct, procentual_correct
