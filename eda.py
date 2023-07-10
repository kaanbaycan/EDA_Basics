def df_problem1(dataframe):
    problematic_cols = []
    for col in dataframe.columns:
        numbers = ["0","1","2","3","4","5","6","7","8","9"]
        if col[0] in numbers :
            problematic_cols.append(col)
    return problematic_cols
    
def unistats(df,sorted):
    import pandas as pd
    pd.set_option("display.max_rows",100)
    pd.set_option("display.max_columns",100)
    output_df = pd.DataFrame(columns = ["Count","Missing","Unique", "Dtype", "Mode", "Mean", "Min", "25%", "Median", "75%", "Max", "Std", "Skew", "Kurt"])
    
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            output_df.loc[col] =[df[col].count() ,df[col].isnull().sum() ,df[col].nunique() ,df[col].dtype ,df[col].mode().values[0], df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75),df[col].max(), df[col].std(), df[col].skew(),df[col].kurt()]   
        else:
            output_df.loc[col] =[df[col].count() ,df[col].isnull().sum() ,df[col].nunique() ,df[col].dtype , "-", "-", "-","-", "-", "-","-", "-", "-","-"]  
        
        
    return output_df.sort_values(by = ["Dtype",sorted])

def correlation(dataframe, target):
    import pandas as pd
    from scipy import stats  
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
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.diagnostic import het_white
    output_df = pd.DataFrame(columns = ["LM_stat", "p_value", "F_stat", "p_value"])
    for col in dataframe:
            if pd.api.types.is_numeric_dtype(dataframe[col]) ==  False:
                model = ols(formula= f"{target}~{col}", data = dataframe).fit()
                breushpagan_test = het_breuschpagan(model.resid,model.model.exog)
                output_df.loc[col] = [breushpagan_test[0],round(breushpagan_test[1],5),breushpagan_test[2],round(breushpagan_test[3],5)]
    return output_df

def anova(dataframe, target):
    import pandas as pd
    import numpy as np
    from scipy import stats
    
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
