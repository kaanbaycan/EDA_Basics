import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from scipy import stats  
import matplotlib.pyplot as plt
import seaborn as sns


def df_problem1(dataframe):
    problematic_cols = []
    for col in dataframe.columns:
        numbers = ["0","1","2","3","4","5","6","7","8","9"]
        if col[0] in numbers :
            problematic_cols.append(col)
    return problematic_cols

def df_problem_solver1(dataframe):
    problematic_cols = []
    for col in dataframe:
        if col[0].isdigit() :
            dataframe.rename(columns={col:"A" + "_" + col}, inplace=True)

def unistats(dataframe,sorted="Missing"):
    pd.set_option("display.max_rows",100)
    pd.set_option("display.max_columns",100)
    output_df = pd.DataFrame(columns = ["Count","Missing","Unique", "Dtype", "Mode", "Mean", "Min", "25%", "Median", "75%", "Max", "Std", "Skew", "Kurt"])
    
    for col in dataframe:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            output_df.loc[col] =[dataframe[col].count() ,dataframe[col].isnull().sum() ,dataframe[col].nunique() ,dataframe[col].dtype ,dataframe[col].mode().values[0], dataframe[col].mean(), dataframe[col].min(), dataframe[col].quantile(0.25), dataframe[col].median(), dataframe[col].quantile(0.75),dataframe[col].max(), dataframe[col].std(), dataframe[col].skew(),dataframe[col].kurt()]   
        else:
            output_df.loc[col] =[dataframe[col].count() ,dataframe[col].isnull().sum() ,dataframe[col].nunique() ,dataframe[col].dtype , "-", "-", "-","-", "-", "-","-", "-", "-","-"]  
        
        
    return output_df.sort_values(by = ["Dtype",sorted])

def correlation(dataframe, target):

    output_dataframe = pd.DataFrame(columns = ["Columns","P-Value","Correlation Coefficient(r)","Absolute r"])
    for col in dataframe.drop(target,axis = 1):
        try:
            if pd.api.types.is_numeric_dtype(dataframe[col]):
                r, p = stats.pearsonr(dataframe[col],dataframe[target])
                output_dataframe.loc[col] = [f"{target}-{col}",round(p,4),r,abs(r)] 
            else:
                pass
        except:
            pass
    return output_dataframe.sort_values(by  = "Absolute r", ascending=False)

def breush_pagan(dataframe,target):
    output_df = pd.DataFrame(columns = ["LM_stat", "p_value", "F_stat", "p_value"])
    for col in dataframe:
            if pd.api.types.is_numeric_dtype(dataframe[col]) ==  False:
                model = ols(formula= f"{target}~{col}", data = dataframe).fit()
                breushpagan_test = het_breuschpagan(model.resid,model.model.exog)
                output_df.loc[col] = [breushpagan_test[0],round(breushpagan_test[1],5),breushpagan_test[2],round(breushpagan_test[3],5)]
    return output_df

def anova(dataframe, target):
    
    output_df = pd.DataFrame(columns = ["F_stat", "p_value~"])    
    for col in dataframe:
        if pd.api.types.is_numeric_dtype(dataframe[col]) == False:
            categories = dataframe[col].unique()     
            if len(categories) >= 2:
                cat_val = []
                for cat in categories:
                    cat_val.append(dataframe[dataframe[col] == cat][target])
  
            
            f,p = stats.f_oneway(*cat_val)
            output_df.loc[col] = [f,round(p,6)]


   
    return output_df.sort_values(by = "F_stat", ascending=False)

def scatter(dataframe, target, feature):

    sns.set_style(style="white")
        
    model = ols(formula= f"{target}~{feature}", data = dataframe).fit()
        
    lm, p1, f, p2 = het_breuschpagan(model.resid,model.model.exog)
    m, b, r, p, err = stats.linregress(dataframe[feature], dataframe[target])
    
    string = "y = " + str(round(m,2)) + "x" + str(round(b,2)) + "\n"
    string += "r_2 = " + str(round(r**2, 4)) +"\n"
    string += "p = " + str(round(p, 5)) + "\n"
    string += str(dataframe[feature].name) + " skew = " + str(round(dataframe[feature].skew(), 2)) + "\n"
    string += str(dataframe[target].name) + " skew = " + str(round(dataframe[target].skew(), 2)) + "\n"
    string += str(dataframe[feature].name) + " Breushpagan Test = " + "LM stat: " + str(round(lm,4)) + " p value: " + str(round(p1,4)) + " F stat: " + str(round(f,4)) + " p value: " + str(round(p2,4)) + "\n"                                                                                 
    ax = sns.jointplot(x = feature, y = target, kind = "reg", data = dataframe)
    ax.fig.text( 1, 0.1, string, fontsize = 12, transform = plt.gcf().transFigure)
    
def barplots(dataframe, label, target):
    ttests = []
    string = "Categories  t stat  p value \n "
    if pd.api.types.is_numeric_dtype(dataframe[label]) == False and len(dataframe[label].unique()) <= 15 and len(dataframe[label].unique())>=2:
        e_types = dataframe[label].unique()
        for i, e in enumerate(e_types):
            for i2, e2 in enumerate(e_types):
                if i2 >= i :
                    g1 = dataframe[dataframe[label] == e][target]
                    g2 = dataframe[dataframe[label] == e2][target]
                    t, p = stats.ttest_ind(g1,g2)
                    ttests.append([f"{e} - {e2}", t, p])
                    string += (f"{e} - {e2}: {t:.4f}, {p:.5f} " + " \n ")
        plt.figure()
        plt
        viz = sns.barplot(data = dataframe, x = label , y = target,)
        viz.set_xticklabels(viz.get_xticklabels(), rotation = 90)
        viz.set(title = f"{target} by {label}")
        plt.text(5, 0.1, string, fontsize = 12)

def histogram(dataframe,feature, target):
    groups = dataframe[feature].unique()
    grouped_values=[]
    for group in groups:
        grouped_values.append(dataframe[dataframe[feature]== group][target])
    if len(grouped_values) <= 15 and len(grouped_values) >= 2:
        f,p = stats.f_oneway(*grouped_values)
        plt.figure()
        string = "F statistics = " + str(round(f,4)) +"\n"+  "P value = " + str(round(p,4)) + "\n"                                                                                
        ax =  sns.histplot(dataframe, x = target, hue = feature, kde =True)
        ax.text( 1, 0.1, string, fontsize = 12, transform = plt.gcf().transFigure)
        plt.show()

def biv_stats(dataframe, target):
    for col in dataframe:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            ed.scatter(dataframe,col,target)
    for col in dataframe:
        ed.histogram(dataframe,col,target)
    for col in dataframe:
        ed.barplots(dataframe,col,target)

def cleaner(dataframe):
    import eda as ed
    dataframe.dropna(axis=1, inplace=True)
    dataframe.columns = dataframe.columns.str.strip()
    ed.df_problem_solver1(dataframe)
    return dataframe

def dummy(dataframe):
    for col in dataframe:
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            dataframe = dataframe.join(pd.get_dummies(dataframe[col], prefix = col, drop_first = True))
    dataframe = dataframe.select_dtypes(np.number)
    return dataframe

def regression(dataframe,target):   
    y = dataframe[target]
    x = dataframe.drop(target, axis = 1).assign(const=1)
    results = sm.OLS(y,x).fit()
    return results

def regression_stats_df(results):
    df_stats = pd.DataFrame({"coef":results.params, "t":abs(results.tvalues), "p":results.pvalues})
    #df_stats.drop("const", axis=1, inplace = True)
    df_stats = df_stats.sort_values(by = ["t","p"])
    return df_stats
