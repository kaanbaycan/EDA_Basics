#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px


# # Scalar Columns

# In[2]:


def statistics(dataframe):
    #for i in dataframe:
    #    print(f"Number of rows in {i}: {dataframe[i].count()}")

    for i in dataframe:
        print(f"Number of unique values in {i}: {dataframe[i].nunique()}")
    
    #for i in dataframe:
      #  print(f"Data type of {i}: {dataframe[i].dtype}")
        
    for i in dataframe:
        if pd.api.types.is_numeric_dtype(dataframe[i]):
            print(f"Data type of {i} is numeric and {dataframe[i].dtype}")
    
    for i in dataframe:
        if dataframe[i].isnull().sum() > 0:
            print(f"Number of null values in {i}: {dataframe[i].isnull().sum()}")
    
    for i in dataframe:
        if pd.api.types.is_numeric_dtype(dataframe[i]):
            print(f"Kurtosis coeff of {i}: {kurtosis(dataframe[i], bias = False)}")
    for i in dataframe:
        if pd.api.types.is_numeric_dtype(dataframe[i]):
            print(f"Skewness coeff of {i}: {skew(dataframe[i], bias=False)}")


# In[3]:


def correlation(dataframe,column1,column2):
    dataframe2 = dataframe.dropna(subset = [column1,column2])
    r, p = stats.pearsonr(dataframe2[column1],dataframe2[column2])
    print(f"Correleation between {column1} and {column2} is : {r:.4f}")
    print(f"P-value of this correleation is: {p:.4f}")


# In[4]:


def F_stat(dataframe,column1,column2):
    model = ols(formula= f"{column1}~{column2}", data = dataframe).fit()
    
    white_test = het_white(model.resid, model.model.exog)
    breushpagan_test = het_breuschpagan(model.resid,model.model.exog)
    
    output_df = pd.DataFrame(columns = ["LM stat", "LM p value", "F stat", "F stat p value"])
    output_df.loc["white"] = white_test
    output_df.loc["breushpagan"] = breushpagan_test
    return output_df


# In[5]:


def three_d_plot(dataframe,columns=[],color=None ,symbol=None, size=8): 
    """columns[4] will be the colored"""
    font = {"size" : size} 
    plt.rc("font", **font)

    fig = plt.figure()
    three_d_plot = Axes3D(fig)
    three_d_plot.scatter(df[columns[0]], df[columns[1]], df[columns[2]])
    
    fig = px.scatter_3d(dataframe, x = columns[0], y = columns[1], z = columns[2], color = color, symbol = symbol)
    

    return plt.show(), fig.show()


# In[129]:


def anova(dataframe,feature, target):
    groups = dataframe[feature].unique()
    grouped_values=[]
    for group in groups:
        grouped_values.append(dataframe[dataframe[feature]== group][label])

    grouped_values
    f,p = stats.f_oneway(*grouped_values)
    print(f"f-value: {f}") #large t-values presents a strong relation between categories
    print(f"p-value: {p:.6f}") #small p-values(less than 0.05 or 0.01) is enough to reject the null hypothesis
    plt.figure(figsize=[16,9])
    sns.histplot(dataframe, x = target, hue = feature,kde =True)
    ("")


# In[10]:


#with the jointplot histograms also occur
sns.set_style(style="white")
sns.jointplot(x = "LotFrontage", y = "SalePrice", data= df, kind="hex");


# In[9]:


df = pd.read_csv("train.csv")
df.dropna()


# # Categorical Data 

# In[12]:


statistics(df)


# In[13]:


df


# In[14]:


df.MSZoning.value_counts()


# In[15]:


RL = df.SalePrice[df.MSZoning == "RL"]
RM = df.SalePrice[df.MSZoning == "RM"]
FV = df.SalePrice[df.MSZoning == "FV"]
RH = df.SalePrice[df.MSZoning == "RH"]
C = df.SalePrice[df.MSZoning == "C (all)"]


# In[16]:


#conducting a t-test for two different categories

t, p = stats.ttest_ind(RM,FV)

print(f"t-value: {t}") #large t-values presents a strong relation between categories
print(f"p-value: {p:.6f}") #small p-values(less than 0.05 or 0.01) is enough to reject the null hypothesis


# In[17]:


sns.set(style = "darkgrid")
plt.figure(figsize=[16,9])
sns.histplot(data = RL, color = "red", label = "RL", kde = True)
sns.histplot(data = RM, color = "skyblue", label = "RM", kde = True)
sns.histplot(data = RH, color = "purple", label = "RH", kde = True)
sns.histplot(data = FV, color = "green", label = "FV", kde = True)
sns.histplot(data = C, color = "orange", label = "C", kde = True)
plt.legend()
("")


# In[18]:


#comparison of more than 2 groups variances accros the means
f, p = stats.f_oneway(RM,FV)
print(f"f-value: {f}") #large t-values presents a strong relation between categories
print(f"p-value: {p:.6f}") #small p-values(less than 0.05 or 0.01) is enough to reject the null hypothesis


# In[19]:


plt.figure(figsize=[16,9])
sns.histplot(df, x = "SalePrice", hue = "OverallCond",kde =True);


# In[20]:


t, p = stats.ttest_ind(df.SalePrice[df.OverallCond == 9],df.SalePrice[df.OverallCond == 2])
print(f"t-value: {t}") #large t-values presents a strong relation between categories
print(f"p-value: {p:.6f}") #small p-values(less than 0.05 or 0.01) is enough to reject the null hypothesis


# In[21]:


def anova(dataframe,feature, target):
    groups = dataframe[feature].unique()
    grouped_values=[]
    for group in groups:
        grouped_values.append(dataframe[dataframe[feature]== group][label])

    grouped_values
    f,p = stats.f_oneway(*grouped_values)
    print(f"f-value: {f}") #large t-values presents a strong relation between categories
    print(f"p-value: {p:.6f}") #small p-values(less than 0.05 or 0.01) is enough to reject the null hypothesis
    plt.figure(figsize=[16,9])
    sns.histplot(dataframe, x = target, hue = feature,kde =True);


# In[26]:


plt.figure(figsize=(10,5))
sns.barplot(data = df, x = "OverallQual" , y = "SalePrice",);


# In[31]:


#to change the x labels rotation
viz = sns.barplot(data = df, x = "OverallQual" , y = "SalePrice",)
viz.set_xticklabels(viz.get_xticklabels(), rotation = 90);


# In[56]:


#ordering and changing the estimator
viz = sns.barplot(data = df, x = "OverallQual" , y = "SalePrice", 
                  estimator= np.mean, #related to bars, 
                  ci="sd",#ci is related to line
                  order=[1,2,3,4,5,6,7,8,9,10],
                  palette="Blues_d",
                  hue = "MSZoning" #adding another classification by a column
                 )
                  
viz.set_xticklabels(viz.get_xticklabels(), rotation = 90);


# In[67]:


#ordering and changing the estimator

viz = sns.catplot(data = df, x = "OverallQual" , y = "SalePrice", 
                  estimator= np.mean, #related to bars, 
                  ci="sd",#ci is related to line
                  order=[1,2,3,4,5,6,7,8,9,10],
                  palette="Blues_d",
                  hue = "MSZoning", #adding another classification by a column
                  col = "LotShape") #adding one mor cluster

viz.set_xticklabels( rotation = 90);


# In[80]:


viz = sns.catplot(data = df, x = "OverallQual" , y = "SalePrice", 
                  estimator= np.mean, #related to bars, 
                  ci="sd",#ci is related to line
                  order=[1,2,3,4,5,6,7,8,9,10],
                  palette="Blues_d",
                  #hue = "MSZoning", #adding another classification by a column
                  #col = "LotShape",
                  kind = "swarm") #changing the type of plot

viz.set_xticklabels( rotation = 0);


# In[81]:


#GOOD FOR FİNANCE DATA
x = ['A', 'B', 'C', 'D']
y1 = np.array([10, 20, 10, 30])
y2 = np.array([20, 25, 15, 25])
y3 = np.array([12, 15, 19, 6])
y4 = np.array([10, 29, 13, 19])
 
# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='y')
plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Teams")
plt.ylabel("Score")
plt.legend(["Round 1", "Round 2", "Round 3", "Round 4"])
plt.title("Scores by Teams in 4 Rounds")
plt.show()


# In[85]:


#pairwise analysis of variance(anova)
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(df["OverallCond"], df["OverallQual"])
print(mc.tukeyhsd())


# In[167]:


def pairwise_ttest(dataframe, label, target):
    e_types = dataframe[label].unique()
    ttests = []
    for i, e in enumerate(e_types):
        for i2, e2 in enumerate(e_types):
            if i2 > i :
                g1 = dataframe[df[label] == e][target]
                g2 = dataframe[df[label] == e2][target]
                t, p = stats.ttest_ind(g1,g2)
                ttests.append([f"{e} - {e2}", t, p])
    significant_ttest = []
    for test in ttests:
        if test[2] <= 0.05/len(ttests):
            significant_ttest.append(test)
    return significant_ttest


# In[173]:


pairwise_ttest(df, "OverallQual", "SalePrice")


# In[174]:


anova(df, "OverallQual","SalePrice")


# In[ ]:




