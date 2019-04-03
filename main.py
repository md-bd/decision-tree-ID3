
import pandas as pd
import numpy as np
import functions as f


# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
filename = 'car.data.csv'
col_name_list = ['buying','maint','doors','persons','lug_boot','safety','class']

ds = pd.read_csv(filename, header=None, names=col_name_list)

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
ds = ds.sample(frac=1).reset_index(drop=True)
# print(ds.head(10))
# print(len(ds))
training_last_index = round(len(ds) * 0.8)

training_data_set = ds.iloc[:training_last_index].reset_index(drop=True)
testing_data_set = ds.iloc[training_last_index:].reset_index(drop=True)


total_col = list(training_data_set.columns)
# print(total_col)
target_col = total_col[len(total_col) - 1]

at = total_col[:len(total_col) - 1]

tree = f.ID3(training_data_set, training_data_set, target_col, at)
# print('---Dicision Tree---\n', tree)


# ----------------Prediction and accuracy----------------------
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#selection-by-position
# http://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.to_dict.html
query = testing_data_set.iloc[:, :-1].to_dict('index')

predicted = pd.DataFrame(columns=["predicted"])

# predict every testing data one by one
for i in query:
    # print(query[i])
    pre = f.predict(query[i], tree)
    # print(pre)
    predicted.loc[i, "predicted"] = pre

# https://datatofish.com/export-dataframe-to-csv/
# export_csv = predicted.to_csv(r'export_predicted_dataframe.csv', index=None, header=True)
print('Total Predictions: ', len(predicted))

"""
print('---predicted results---')
print((predicted["predicted"]).head(10))
print("---Actual results---")
print((testing_data_set[target_col]).head(10))
"""
print('----wrong predictions----')
wrong_d = pd.DataFrame(data = testing_data_set[predicted["predicted"] != testing_data_set[target_col]])
wrong_d["predicted"] = predicted[predicted["predicted"] != testing_data_set[target_col]]
print(wrong_d)
print("Total Wrong Predictions: ", len(wrong_d))
# print(testing_data_set[predicted["predicted"] != testing_data_set[target_col]])
# print(len(predicted[predicted["predicted"] != testing_data_set[target_col]]))

print('***********************\t \t*********')
print('The prediction accuracy is: ', round((np.sum(predicted["predicted"] == testing_data_set[target_col])/len(testing_data_set))*100, 2), '%')
print('***********************\t \t*********')
