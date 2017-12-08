
# coding: utf-8

# In[8]:


## Classification using CART algorithm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score

# read the dataset
dataset = pd.read_table(filepath_or_buffer = '..\dataset\letter.data',sep='\t', header = None);

# the dataset had a tab space at the end of the line hence producing NaN values, Discard that column
dataset = dataset.iloc[:,0:134];

# label the columns of the dataset
names=['Id','letter','next_id','word_id','position','fold','p_0_0','p_0_1','p_0_2','p_0_3','p_0_4','p_0_5','p_0_6','p_0_7','p_1_0','p_1_1','p_1_2','p_1_3','p_1_4','p_1_5','p_1_6','p_1_7','p_2_0','p_2_1','p_2_2','p_2_3','p_2_4','p_2_5','p_2_6','p_2_7','p_3_0','p_3_1','p_3_2','p_3_3','p_3_4','p_3_5','p_3_6','p_3_7','p_4_0','p_4_1','p_4_2','p_4_3','p_4_4','p_4_5','p_4_6','p_4_7','p_5_0','p_5_1','p_5_2','p_5_3','p_5_4','p_5_5','p_5_6','p_5_7','p_6_0','p_6_1','p_6_2','p_6_3','p_6_4','p_6_5','p_6_6','p_6_7','p_7_0','p_7_1','p_7_2','p_7_3','p_7_4','p_7_5','p_7_6','p_7_7','p_8_0','p_8_1','p_8_2','p_8_3','p_8_4','p_8_5','p_8_6','p_8_7','p_9_0','p_9_1','p_9_2','p_9_3','p_9_4','p_9_5','p_9_6','p_9_7','p_10_0','p_10_1','p_10_2','p_10_3','p_10_4','p_10_5','p_10_6','p_10_7','p_11_0','p_11_1','p_11_2','p_11_3','p_11_4','p_11_5','p_11_6','p_11_7','p_12_0','p_12_1','p_12_2','p_12_3','p_12_4','p_12_5','p_12_6','p_12_7','p_13_0','p_13_1','p_13_2','p_13_3','p_13_4','p_13_5','p_13_6','p_13_7','p_14_0','p_14_1','p_14_2','p_14_3','p_14_4','p_14_5','p_14_6','p_14_7','p_15_0','p_15_1','p_15_2','p_15_3','p_15_4','p_15_5','p_15_6','p_15_7'];
dataset.columns = names;

# Drop unnecessary columns that are not relevant to this problem
dataset = dataset.drop(labels=['Id','next_id','word_id', 'position'], axis=1)
dataset

# how many datapoints each fold has 
dataset.fold.value_counts()

y = dataset.letter
X = dataset.iloc[:, 2:131]

# Now that we have the features and the target, split them into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Initialize the classifier
clf = tree.DecisionTreeClassifier(random_state=0)
scores = cross_val_score(clf, X, y, cv=10);
print(scores)
print(np.mean(scores))

