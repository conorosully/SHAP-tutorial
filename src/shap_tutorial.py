#!/usr/bin/env python
# coding: utf-8

# # SHAP Tutorial 
# <br>
# Course sections:
# <ol>
# <li> SHAP values
# <li> SHAP aggregations
#     <ol>
#     <li> Force plots
#     <li> Mean SHAP
#     <li> Beeswarm
#     <li> Violin
#     <li> Heatmap
#     <li> Dependence
#     </ol>
# <li> Custom SHAP plots
# <li> Binary and categorical target variables 
# <li> SHAP interaction values
# <li> Categorical features
# </ol>
# <br>
# <b>Dataset:</b> https://archive.ics.uci.edu/ml/datasets/Abalone

#imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

import shap
shap.initjs()


# # Dataset

#import dataset
data = pd.read_csv("../data/abalone.data",
                  names=["sex","length","diameter","height",
                         "whole weight","shucked weight",
                         "viscera weight","shell weight",
                         "rings"])

print(len(data))
data.head()


#Plot 1: whole weight
plt.scatter(data['whole weight'],data['rings'])
plt.ylabel('rings',size=20)
plt.xlabel('whole weight',size=20)


#Plot 2: sex
plt.boxplot(data[data.sex=='I']['rings'],positions=[1])
plt.boxplot(data[data.sex=='M']['rings'],positions=[2])
plt.boxplot(data[data.sex=='F']['rings'],positions=[3]) 

plt.xticks(ticks=[1,2,3],labels=['I', 'M', 'F'],size= 15)
plt.ylabel('rings',size=20)
plt.xlabel('sex',size=20)


#Plot 3: Correlation heatmap
cont = ["length","diameter","height",
        "whole weight","shucked weight",
        "viscera weight","shell weight",
        "rings"]
corr_matrix = pd.DataFrame(data[cont],columns=cont).corr()

sns.heatmap(corr_matrix,
            cmap='coolwarm',
            center = 0,
            annot=True,
            fmt='.1g')


# # Feature Engineering

y = data['rings']
X = data[["sex","length","height",
          "shucked weight","viscera weight","shell weight"]]


#Create dummy variables
X['sex.M'] = [1 if s == 'M' else 0 for s in X['sex']]
X['sex.F'] = [1 if s == 'F' else 0 for s in X['sex']]
X['sex.I'] = [1 if s == 'I' else 0 for s in X['sex']]
X = X.drop('sex', axis=1)

X.head()


# # Modelling

#Train model
model = xgb.XGBRegressor(objective="reg:squarederror") 
model.fit(X, y)


#Get predictions
y_pred = model.predict(X)

#Model evaluation
plt.figure(figsize=(5, 5))

plt.scatter(y,y_pred)
plt.plot([0, 30], 
         [0, 30], 
         color='r', 
         linestyle='-', 
         linewidth=2)

plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)


# # 1) Standard SHAP values

#Get shap values
explainer = shap.Explainer(model)
shap_values = explainer(X)

#shap_values = explainer(X[0:100])


np.shape(shap_values.values)


# ## Waterfall plot

# Waterfall plot for first observation

#set face color white
shap.plots.waterfall(shap_values[0])


# Waterfall plot for first observation
shap.plots.waterfall(shap_values[1], max_display=4)


# # 2) SHAP aggregations

# <b>Note:</b> from here on we only consider the continous target variable 

# ## Force plot

shap.plots.force(shap_values[0])


# ## Stacked force plot

shap.plots.force(shap_values[0:100])


# ## Absolute Mean SHAP

shap.plots.bar(shap_values)


# ## Beeswarm plot

shap.plots.beeswarm(shap_values)


# ## Violin plot

# violin plot
shap.plots.violin(shap_values)


# layered violin plot
shap.plots.violin(shap_values, plot_type="layered_violin")


# ## Heamap

# heatmap
shap.plots.heatmap(shap_values)


# order by predictions
order = np.argsort(y_pred)
shap.plots.heatmap(shap_values, instance_order=order)


# order by shell weight value
order = np.argsort(data['shell weight'])
shap.plots.heatmap(shap_values, instance_order=order)


# ## Dependence plots

#Plot 1: shell weight
shap.plots.scatter(shap_values[:,"shell weight"])


shap.plots.scatter(shap_values[:,"shell weight"],
                   color=shap_values[:,"shucked weight"])


#Plot 2: shucked weight
shap.plots.scatter(shap_values[:,"shucked weight"])


# # 3) Custom Plots

#Output SHAP object 
shap_values


np.shape(shap_values.values)


X.head()


# SHAP correlation plot 
corr_matrix = pd.DataFrame(shap_values.values,
                           columns=X.columns).corr()


sns.set(font_scale=1)
sns.heatmap(corr_matrix,
            cmap='coolwarm',
            center = 0, 
            annot=True,
            fmt='.1g')


# # 3.  Binary and categorical target variables 

# ### Binary target variable

#Binary target varibale
y_bin = [1 if y_>10 else 0 for y_ in y]

#Train model 
model_bin = xgb.XGBClassifier(objective="binary:logistic")
model_bin.fit(X, y_bin)

#Get shap values
explainer = shap.Explainer(model_bin)
shap_values_bin = explainer(X)


# waterfall plot for first observation
shap.plots.waterfall(shap_values_bin[0])


# waterfall plot for first observation
shap.plots.bar(shap_values_bin)


# ### Categorical target variables

#Categorical target varibale
y_cat = [2 if y_>12 else 1 if y_>8 else 0 for y_ in y]

#Train model 
model_cat = xgb.XGBClassifier(objective="binary:logistic")
model_cat.fit(X, y_cat)

# get probability predictions
model_cat.predict_proba(X)[0]


#Get shap values
explainer = shap.Explainer(model_cat)
shap_values_cat= explainer(X)

print(np.shape(shap_values_cat))


# waterfall plot for first observation
shap.plots.waterfall(shap_values_cat[0,:,0])

# waterfall plot for first observation
shap.plots.waterfall(shap_values_cat[0,:,1])

# waterfall plot for first observation
shap.plots.waterfall(shap_values_cat[0,:,2])


# Calculate mean SHAP values for each class
mean_0 = np.mean(np.abs(shap_values_cat.values[:,:,0]),axis=0)
mean_1 = np.mean(np.abs(shap_values_cat.values[:,:,1]),axis=0)
mean_2 = np.mean(np.abs(shap_values_cat.values[:,:,2]),axis=0)

df = pd.DataFrame({'small':mean_0,'medium':mean_1,'large':mean_2})

# Plot mean SHAP values
fig,ax = plt.subplots(1,1,figsize=(20,10))
df.plot.bar(ax=ax)

ax.set_ylabel('Mean SHAP',size = 30)
ax.set_xticklabels(X.columns,rotation=45,size=20)
ax.legend(fontsize=30)


#Get model predictions
preds = model_cat.predict(X)

new_shap_values = []
for i,pred in enumerate(preds):
    # Get shap values for predicted class
    new_shap_values.append(shap_values_cat.values[i][:,pred])

#replace shap values
shap_values_cat.values = np.array(new_shap_values)
print(shap_values_cat.shape)


# plot the updated SHAP values 
shap.plots.bar(shap_values_cat)


shap.plots.beeswarm(shap_values_cat)


# # 4) SHAP interaction value

#Get SHAP interaction values
explainer = shap.Explainer(model)
shap_interaction = explainer.shap_interaction_values(X)


#Get shape of interaction values
np.shape(shap_interaction)


# SHAP interaction values for first employee
shap_0 = np.round(shap_interaction[0],2)
pd.DataFrame(shap_0,
             index=X.columns,
             columns=X.columns)


# ## Mean SHAP interaction values

# Get absolute mean of matrices
mean_shap = np.abs(shap_interaction).mean(0)
mean_shap = np.round(mean_shap,1)

df = pd.DataFrame(mean_shap,index=X.columns,columns=X.columns)

# times off diagonal by 2
df.where(df.values == np.diagonal(df),df.values*2,inplace=True)

# display 
sns.set(font_scale=1)
sns.heatmap(df,cmap='coolwarm',annot=True)
plt.yticks(rotation=0) 


# ## Dependence plot

shap.dependence_plot(
    ("shell weight", "shucked weight"),
    shap_interaction, X,
    display_features=X)


#Interaction between shell weight and shucked weight
plt.scatter(data["shell weight"],
            data["shucked weight"],
            c=data["rings"],
            cmap='bwr')
plt.colorbar(label="Number of Rings", 
             orientation="vertical")

plt.xlabel('shucked weight',size=15)
plt.ylabel('shell weight',size=15)


# # 5) SHAP for categorical variables

X.head()


# Waterfall plot for first observation
shap.plots.waterfall(shap_values[0])


new_shap_values = []

#loop over all shap values:
for values in shap_values.values:
    
    #sum SHAP values for sex 
    sv = list(values)
    sv = sv[0:5] + [sum(sv[5:8])]
    
    new_shap_values.append(sv)


#replace shap values
shap_values.values = np.array(new_shap_values)

#replace data with categorical feature values 
X_cat = data[["length","height",
              "shucked weight","viscera weight",
              "shell weight","sex"]]
shap_values.data = np.array(X_cat)

#update feature names
shap_values.feature_names = list(X_cat.columns)


shap.plots.waterfall(shap_values[0])


shap.plots.bar(shap_values)


shap.plots.beeswarm(shap_values)


#get shaply values and data
sex_values = shap_values[:,"sex"].values
sex_data = shap_values[:,"sex"].data
sex_categories = ['I','M','F']

#split sex shap values based on category
sex_groups = []
for s in sex_categories:
    relevant_values = sex_values[sex_data == s]
    sex_groups.append(relevant_values)
    
#plot boxplot
plt.boxplot(sex_groups,labels=sex_categories)

plt.ylabel('SHAP values',size=15)
plt.xlabel('Sex',size=15)


#Create for placeholder SHAP values
shap_values_sex = explainer(X)

#get shaply values and data
sex_values = shap_values[:,"sex"].values
sex_data = shap_values[:,"sex"].data
sex_categories = ['I','M','F']

#Create new SHAP values array

#Split odor SHAP values by unique odor categories
new_shap_values = [np.array(pd.Series(sex_values)[sex_data==s]) 
                    for s in sex_categories]

#Each sublist needs to be the same length
max_len = max([len(v) for v in new_shap_values])
new_shap_values = [np.append(vs,[np.nan]*(max_len - len(vs))) for vs in new_shap_values]
new_shap_values = np.array(new_shap_values)

#transpost matrix so categories are columns and SHAP values are rows
new_shap_values = new_shap_values.transpose()

#replace shap values
shap_values_sex.values = np.array(new_shap_values)

#replace data with placeholder array
shap_values_sex.data = np.array([[0]*len(sex_categories)]*max_len)

#replace base data with placeholder array
shap_values_sex.base = np.array([0]*max_len)

#replace feature names with category labels
shap_values_sex.feature_names = list(sex_categories)

#Use beeswarm as before
shap.plots.beeswarm(shap_values_sex,color_bar=False)


import warnings
warnings.filterwarnings('ignore')

