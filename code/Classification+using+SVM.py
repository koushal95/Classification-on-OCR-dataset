
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np

# read the dataset
# The dataset file is in the dataset folder. 
dataset = pd.read_table(filepath_or_buffer = '..\dataset\letter.data',sep='\t', header = None);

# Display the dataset
dataset



# In[12]:


# the dataset had a tab space at the end of the line hence producing NaN values, Discard that column
dataset = dataset.iloc[:,0:134];

# Now view the dataset
dataset


# In[13]:


# label the columns of the dataset
names=['Id','letter','next_id','word_id','position','fold','p_0_0','p_0_1','p_0_2','p_0_3','p_0_4','p_0_5','p_0_6','p_0_7','p_1_0','p_1_1','p_1_2','p_1_3','p_1_4','p_1_5','p_1_6','p_1_7','p_2_0','p_2_1','p_2_2','p_2_3','p_2_4','p_2_5','p_2_6','p_2_7','p_3_0','p_3_1','p_3_2','p_3_3','p_3_4','p_3_5','p_3_6','p_3_7','p_4_0','p_4_1','p_4_2','p_4_3','p_4_4','p_4_5','p_4_6','p_4_7','p_5_0','p_5_1','p_5_2','p_5_3','p_5_4','p_5_5','p_5_6','p_5_7','p_6_0','p_6_1','p_6_2','p_6_3','p_6_4','p_6_5','p_6_6','p_6_7','p_7_0','p_7_1','p_7_2','p_7_3','p_7_4','p_7_5','p_7_6','p_7_7','p_8_0','p_8_1','p_8_2','p_8_3','p_8_4','p_8_5','p_8_6','p_8_7','p_9_0','p_9_1','p_9_2','p_9_3','p_9_4','p_9_5','p_9_6','p_9_7','p_10_0','p_10_1','p_10_2','p_10_3','p_10_4','p_10_5','p_10_6','p_10_7','p_11_0','p_11_1','p_11_2','p_11_3','p_11_4','p_11_5','p_11_6','p_11_7','p_12_0','p_12_1','p_12_2','p_12_3','p_12_4','p_12_5','p_12_6','p_12_7','p_13_0','p_13_1','p_13_2','p_13_3','p_13_4','p_13_5','p_13_6','p_13_7','p_14_0','p_14_1','p_14_2','p_14_3','p_14_4','p_14_5','p_14_6','p_14_7','p_15_0','p_15_1','p_15_2','p_15_3','p_15_4','p_15_5','p_15_6','p_15_7'];
dataset.columns = names;

# View the dataset
dataset


# In[4]:


## Exploratory Data Analysis
dataset.shape


# In[5]:


# Some statistics on the table
# May not be of much help as the data is binary 
dataset.describe()


# In[6]:


# check the Datatypes
dataset.dtypes


# In[7]:


# distribution of each letter or The number of rows for each letter
dataset.letter.value_counts()


# In[8]:


# check if there are any missing values in the dataset
dataset.isnull().values.any()


# In[14]:


# Check the working of fold attribute (Does each value of fold have all the letters a to z?)
# Check the effectiveness of fold attribute
table = pd.pivot_table(dataset, index=['fold'], columns = ['letter'], values='Id', aggfunc=np.count_nonzero)

# view the table
table


# In[15]:


# check if a fold has no letter in it
table.isnull().values.any()


# In[16]:


# the fold attribute has all letters in each of its fold value


# In[17]:


## Now we need to train and test 
# we have the following ways:
    # 1) Using fold attribute to split the dataset into training and test set
    # 2) Using train_test_split method from library to generate the splits
    # 3) Implementing cross validation by using croos_val_score method from the library
# Note that cross validation takes time.

# Approach 1: Using fold attribute

# drop unnecessary featues
dataset = dataset.drop(labels=['next_id','word_id', 'position'], axis=1)

# Split dataset into training and testing using folds (8 for training 2 for testing)
folds = dataset.fold.unique()
folds 


# In[18]:


folds = np.sort(folds)
folds


# In[19]:


# assign the folds 1 to 8 to train data
appended_data = []
for i in folds[1:9]:
    appended_data.append(dataset[dataset.fold == i])

train_data = pd.concat(appended_data, axis = 0)

# view train data
train_data


# In[20]:


#assign the folds 0 and 9 to test data
appended_data = []
appended_data.append(dataset[dataset.fold == 0])
appended_data.append(dataset[dataset.fold == 9])
test_data = pd.concat(appended_data, axis = 0)

# view test data
test_data


# In[21]:


# Now divide the dataset into features and target
train_X = train_data.iloc[:,3:131]

# view train_X
train_X


# In[22]:


# Assigning target features
train_y = train_data.letter

# view train_y
train_y


# In[23]:


# Now assign test_X
test_X = test_data.iloc[:,3:131]

# View the data
test_X


# In[24]:


test_y = test_data.letter

# view the data
test_y


# In[27]:


## Approach 1: Using fold attribute
from sklearn.svm import SVC
from sklearn import metrics

# initialize the classifier
clf = SVC()

# train the classifier
clf.fit(train_X, train_y)

# test the classifier 
predictions = clf.predict(test_X)

# report the classifier accuracy
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(test_y, predictions)))


# In[29]:


## Approach 2: Using train_test_split from the library
from sklearn.model_selection import train_test_split

# I am reading the dataset again just to maintain clarity between the approaches
# read the dataset
dataset = pd.read_table(filepath_or_buffer = '..\dataset\letter.data',sep='\t', header = None);
# the dataset had a tab space at the end of the line hence producing NaN values, Discard that column
dataset = dataset.iloc[:,0:134];
# label the columns of the dataset
names=['Id','letter','next_id','word_id','position','fold','p_0_0','p_0_1','p_0_2','p_0_3','p_0_4','p_0_5','p_0_6','p_0_7','p_1_0','p_1_1','p_1_2','p_1_3','p_1_4','p_1_5','p_1_6','p_1_7','p_2_0','p_2_1','p_2_2','p_2_3','p_2_4','p_2_5','p_2_6','p_2_7','p_3_0','p_3_1','p_3_2','p_3_3','p_3_4','p_3_5','p_3_6','p_3_7','p_4_0','p_4_1','p_4_2','p_4_3','p_4_4','p_4_5','p_4_6','p_4_7','p_5_0','p_5_1','p_5_2','p_5_3','p_5_4','p_5_5','p_5_6','p_5_7','p_6_0','p_6_1','p_6_2','p_6_3','p_6_4','p_6_5','p_6_6','p_6_7','p_7_0','p_7_1','p_7_2','p_7_3','p_7_4','p_7_5','p_7_6','p_7_7','p_8_0','p_8_1','p_8_2','p_8_3','p_8_4','p_8_5','p_8_6','p_8_7','p_9_0','p_9_1','p_9_2','p_9_3','p_9_4','p_9_5','p_9_6','p_9_7','p_10_0','p_10_1','p_10_2','p_10_3','p_10_4','p_10_5','p_10_6','p_10_7','p_11_0','p_11_1','p_11_2','p_11_3','p_11_4','p_11_5','p_11_6','p_11_7','p_12_0','p_12_1','p_12_2','p_12_3','p_12_4','p_12_5','p_12_6','p_12_7','p_13_0','p_13_1','p_13_2','p_13_3','p_13_4','p_13_5','p_13_6','p_13_7','p_14_0','p_14_1','p_14_2','p_14_3','p_14_4','p_14_5','p_14_6','p_14_7','p_15_0','p_15_1','p_15_2','p_15_3','p_15_4','p_15_5','p_15_6','p_15_7'];
dataset.columns = names;
# Drop unnecessary columns that are not relevant to this problem
dataset = dataset.drop(labels=['Id','next_id','word_id', 'position'], axis=1)

# the train_test_split function needs the features as first argument and labels as second argument
# So First dividing the dataset into features and labels
y = dataset.letter
X = dataset.iloc[:, 2:131]

# Now that we have the features and the target, split them into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# Note that I changed the variable names for training and test sets between approaches

# initialize the classifier
clf = SVC()

# train the classifier
clf.fit(X_train, y_train)

# test the classifier 
predictions = clf.predict(X_test)

# report the classifier accuracy
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, predictions)))



# In[30]:


## Approach 3: Implementing cross validation using cross_val_score from library
from sklearn.cross_validation import cross_val_score, cross_val_predict

# read the dataset
dataset = pd.read_table(filepath_or_buffer = '..\dataset\letter.data',sep='\t', header = None);
# the dataset had a tab space at the end of the line hence producing NaN values, Discard that column
dataset = dataset.iloc[:,0:134];
# label the columns of the dataset
names=['Id','letter','next_id','word_id','position','fold','p_0_0','p_0_1','p_0_2','p_0_3','p_0_4','p_0_5','p_0_6','p_0_7','p_1_0','p_1_1','p_1_2','p_1_3','p_1_4','p_1_5','p_1_6','p_1_7','p_2_0','p_2_1','p_2_2','p_2_3','p_2_4','p_2_5','p_2_6','p_2_7','p_3_0','p_3_1','p_3_2','p_3_3','p_3_4','p_3_5','p_3_6','p_3_7','p_4_0','p_4_1','p_4_2','p_4_3','p_4_4','p_4_5','p_4_6','p_4_7','p_5_0','p_5_1','p_5_2','p_5_3','p_5_4','p_5_5','p_5_6','p_5_7','p_6_0','p_6_1','p_6_2','p_6_3','p_6_4','p_6_5','p_6_6','p_6_7','p_7_0','p_7_1','p_7_2','p_7_3','p_7_4','p_7_5','p_7_6','p_7_7','p_8_0','p_8_1','p_8_2','p_8_3','p_8_4','p_8_5','p_8_6','p_8_7','p_9_0','p_9_1','p_9_2','p_9_3','p_9_4','p_9_5','p_9_6','p_9_7','p_10_0','p_10_1','p_10_2','p_10_3','p_10_4','p_10_5','p_10_6','p_10_7','p_11_0','p_11_1','p_11_2','p_11_3','p_11_4','p_11_5','p_11_6','p_11_7','p_12_0','p_12_1','p_12_2','p_12_3','p_12_4','p_12_5','p_12_6','p_12_7','p_13_0','p_13_1','p_13_2','p_13_3','p_13_4','p_13_5','p_13_6','p_13_7','p_14_0','p_14_1','p_14_2','p_14_3','p_14_4','p_14_5','p_14_6','p_14_7','p_15_0','p_15_1','p_15_2','p_15_3','p_15_4','p_15_5','p_15_6','p_15_7'];
dataset.columns = names;
# Drop unnecessary columns that are not relevant to this problem
dataset = dataset.drop(labels=['Id','next_id','word_id', 'position'], axis=1)
# Split the dataset into features and 
y = dataset.letter
X = dataset.iloc[:, 2:131]

# initialize the classifier
clf = SVC()

# this will split the data, fit the model, and compute the score for validation set in each iteration (here 10)
# this returns the mean score in each iteration on the validation set, if we want to see what the predictions are we can use cross_val_predict
scores = cross_val_score(clf, X, y, cv = 10)
print ("Cross-validated scores:", scores)



# In[31]:


# Find the mean of the scores of each of the validation set
np.mean(scores)


# In[ ]:


## All the three approaches gave similar results but cross_val_score took more time.

