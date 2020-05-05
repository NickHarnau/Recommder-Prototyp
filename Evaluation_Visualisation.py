import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Evaluation import *


# 1. Different between predicted and true values
#1.1 Predicted and true Values in on graph

# create DF of predicted and true values
Result_df = pd.DataFrame(list(zip(true_values, predicted_values_CB, predicted_values_MF, predicted_values_CF)), columns=["true", "CB", "MF", "CF"])


# all aproaches in one graph
plt.figure(figsize=(10,5))
#plt.plot(Result_df.index, Result_df["true"], "ro", label="true_value")
plt.plot(Result_df.index, Result_df["true"], label="True_Values")
plt.plot(Result_df.index, Result_df["CB"],"yo", label="CB_prediction")
plt.plot(Result_df.index, Result_df["MF"], "bo", label="MF_prediction")
plt.plot(Result_df.index, Result_df["CF"], "go", label="CF_prediction")
plt.xlabel("Reihenfolge der Testwerte von 1- {}".format(len(true_values)))
plt.xticks(np.arange(min(Result_df.index), max(Result_df.index)+1, 2))
plt.ylabel("Werte")
plt.legend()
plt.show()
# per appraoch one graph
for approach in Result_df.columns[1:]:
    plt.figure(figsize=(10,5))
    plt.plot(Result_df.index, Result_df["true"], label="true_value")
    plt.plot(Result_df.index, Result_df["true"], "ro", label="true_value")
    plt.plot(Result_df.index, Result_df[approach],"go", label="{}_prediction".format(approach))
    plt.xticks(np.arange(min(Result_df.index), max(Result_df.index)+1, 2))
    plt.xlabel("Reihenfolge der Testwerte von 1- {}".format(len(true_values)))
    plt.ylabel("Werte")
    plt.legend()
    plt.title("Vorhegesagte Werte mit {} im Vergleich zu den wahren Werten".format(approach))
    plt.show()

#1.2 only the difference between true value and the prdicted one in graph

difference_CB = np.absolute(np.array(true_values) - np.array(predicted_values_CB))
difference_CF = np.absolute(np.array(true_values) - np.array(predicted_values_CF))
difference_MF = np.absolute(np.array(true_values) - np.array(predicted_values_MF))

plt.figure(figsize=(10, 5))
plt.scatter(true_values, predicted_values_CB,  label="CB")
plt.scatter(true_values, predicted_values_CF,  label="CF")
plt.scatter(true_values, predicted_values_MF,  label="MF")
plt.plot(range(10), range(10), "r", label="perfect_fit")
plt.xticks(np.arange(0, 10,1))
plt.xlabel("true_values")
plt.ylabel("predictions")
plt.legend()
plt.title("Comparison of true and predicted values")
plt.show()

"""
plt.plot(true_values, "bo", label="true_values")
plt.plot(true_values, "b")
plt.plot(difference_CB,"r", label="CB")
#plt.plot(difference_CF,"yo", label="CF")
#plt.plot(difference_MF,"go", label="MF")
plt.legend()
plt.xticks(np.arange(0, len(true_values), 2))
plt.title("Comparison true values and predicted values by different Recommder Approach")
plt.show()
"""


# 2. Compare MSE
MSEs = {"CB": MSE_CB, "CF": MSE_CF, "MF": MSE_MF}
MSEs = dict(sorted(MSEs.items() ,key= operator.itemgetter(1)))
plt.figure(figsize=(10,5))
plt.bar(np.arange(len(MSEs.keys())), MSEs.values(), color=["red", "green", "blue"])
plt.xticks(np.arange(len(MSEs.keys())),MSEs.keys())
plt.title("Comparison MSE different approaches")
plt.show()
