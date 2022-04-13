"""
Basado en el trabajo realizado por:
AMRITA CHATTERJEE, https://www.kaggle.com/code/amritachatterjee09/eda-bank-loan-default-risk-analysis/notebook
Posted on September 1, 2020 by George Pipis in Data science , https://python-bloggers.com/2020/09/how-to-run-chi-square-test-in-python/
https://github.com/DavidReveloLuna/Machine-Learning/blob/master/3_4_BosquesAleatorios.ipynb
https://realpython.com/train-test-split-python-data/
https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import itertools
import warnings
import missingno as mn
import time
import openpyxl
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
start_time = time.time()

# setting up plot style 
style.use('seaborn-poster')
style.use('fivethirtyeight')

warnings.filterwarnings('ignore')
nullcol_40_application=[]
Unwanted_application=[]
#Load data
applicationDF = pd.read_csv(r'data/application_data.csv')
previousDF = pd.read_csv(r'data/previous_application.csv')

#Export data to Excel
def export_data(nullcol_40_application, report_name, column_label):
    df = pd.DataFrame(nullcol_40_application, columns = column_label)
    df.to_excel (r'reports\\'+report_name+'.xlsx', index = False, header=True)
    
#Mostrar las dimensiones
def show_dimension():
    # Database dimension
    print("Database dimension - applicationDF     :",applicationDF.shape)
    print("Database dimension - previousDF        :",previousDF.shape)

    #Database size
    print("Database size - applicationDF          :",applicationDF.size)
    print("Database size - previousDF             :",previousDF.size)

#Data Cleaning & Manipulation
def get_matrix_null(applicationDF):
    mn.matrix(applicationDF)
    plt.title("Valores perdidos")
    plt.ylabel("Valores nulos")
    plt.show()

# % null value in each column
def analys_null_column(applicationDF):
    value_null =  round(applicationDF.isnull().sum() / applicationDF.shape[0] * 100.00,2)
    print('Giskard: Comprobación del valor nulo % de cada columna', value_null)


# Value Calculation
def show_figure_loss_data_null(applicationDF, report_name):
    value_calculate = round((applicationDF.isnull().sum())*100/applicationDF.shape[0], 2)
    null_applicationDF = pd.DataFrame(value_calculate).reset_index()
    null_applicationDF.columns = ['Nombre de la columna', 'Valores nulos Porcentaje']
    fig = plt.figure(figsize=(18,6))
    ax = sns.pointplot(x="Nombre de la columna",y="Valores nulos Porcentaje",data=null_applicationDF,color='blue')
    plt.xticks(rotation =90,fontsize =7)
    ax.axhline(40, ls='--',color='red')
    plt.title("Porcentaje de valores perdidos")
    plt.ylabel("Valores nulos Porcentaje")
    plt.xlabel("COLUMNAS")
    plt.show()
    # más o igual al 40% de filas y columnas vacías
    nullcol_40_application = null_applicationDF[null_applicationDF["Valores nulos Porcentaje"]>=40]
    export_data(nullcol_40_application, report_name, ['Nombre de la columna', 'Valores nulos Porcentaje'])

#Crear grafica para ver la correclacion de datos
def get_correlation_columns(applicationDF):
    
    Source = applicationDF
    source_corr = Source.corr()
    ax = sns.heatmap(source_corr,
                xticklabels=source_corr.columns,
                yticklabels=source_corr.columns,
                annot = True,
                cmap ="RdYlGn")
    plt.show()


def get_column_delete(column_correlation):
    Unwanted_application=[]
    column_correlation.remove('TARGET') 
    Unwanted_application = Unwanted_application + column_correlation
    # print('Giskard: columnas que se puede borrar', len(Unwanted_application))

"""
Análisis exploratorio
"""
"""
En base a la matriz anterior, se evidencia que el conjunto de datos tiene muchos valores perdidos.
"""
#get_matrix_null(applicationDF)
#get_matrix_null(previousDF)
"""
Comprobemos para cada columna cuál es el % de valores perdidos
"""
# show_figure_loss_data_null(applicationDF, "report_loss_appliaction")
# show_figure_loss_data_null(previousDF, "report_loss_preview")
"""
Analizar y eliminar columnas innecesarias, verificar si existe correlación
"""
column_correlation = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','TARGET']


# get_correlation_columns(applicationDF[column_correlation])
"""
Si No hay correlación entre columnas selecionadas, or lo tanto, estas columnas pueden suprimirse
"""
get_column_delete(column_correlation)
# Dropping the unnecessary columns from applicationDF
applicationDF.drop(labels=Unwanted_application,axis=1,inplace=True)
# Inspecting the dataframe after removal of unnecessary columns
# print("Database dimension - applicationDF     :",applicationDF.shape)


"""
Normalizacion de datos
"""

# Converting Negative days to positive days

date_col = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']

for col in date_col:
    applicationDF[col] = pd.to_numeric(applicationDF[col], errors='coerce')
    applicationDF[col] = abs(applicationDF[col])

#Conversion of Object and Numerical columns to Categorical Columns
categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY'
                      ]
for col in categorical_columns:
    applicationDF[col] =pd.Categorical(applicationDF[col])

# Creating bins for Age
applicationDF['AGE'] = applicationDF['DAYS_BIRTH'] // 365
bins = [0,20,30,40,50,100]
slots = ['0-20','20-30','30-40','40-50','50 above']

applicationDF['AGE_GROUP']=pd.cut(applicationDF['AGE'],bins=bins,labels=slots)
applicationDF['AGE_GROUP'].value_counts(normalize=True)*100

# Creating bins for Credit amount
applicationDF['AMT_CREDIT']=applicationDF['AMT_CREDIT']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,100]
slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',
       '800k-900k','900k-1M', '1M Above']

applicationDF['AMT_CREDIT_RANGE']=pd.cut(applicationDF['AMT_CREDIT'],bins=bins,labels=slots)

# Creating bins for Employement Time
applicationDF['YEARS_EMPLOYED'] = applicationDF['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

applicationDF['EMPLOYMENT_YEAR']=pd.cut(applicationDF['YEARS_EMPLOYED'],bins=bins,labels=slots)

# Creating Importe del crédito
applicationDF['AMT_CREDIT']=applicationDF['AMT_CREDIT'] // 100000

bins = [0,1,2,3,4,5,6,7,8,9,10,100]
slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',
       '800k-900k','900k-1M', '1M Above']

applicationDF['AMT_CREDIT_RANGE']=pd.cut(applicationDF['AMT_CREDIT'],bins=bins,labels=slots)

#checking the binning of data and % of data in each category
applicationDF['AMT_CREDIT_RANGE'].value_counts(normalize=True)*100


# """
# Balanceo de los datos con respecto a la variable objectivo, buenos pagadores y malos pagadores
# """
# Imbalance = applicationDF["TARGET"].value_counts().reset_index()

# plt.figure(figsize=(10,4))
# x= ['Buen Pagador','Mal Pagador']
# sns.barplot(x,"TARGET",data = Imbalance,palette= ['g','r'])
# plt.xlabel("Estado de reembolso del préstamo")
# plt.ylabel("Recuento de los reembolsos y de los morosos")
# plt.title("Gráfica de balanceo de la varible objectivo")
# plt.show()

# count_0 = Imbalance.iloc[0]["TARGET"]
# count_1 = Imbalance.iloc[1]["TARGET"]
# count_0_perc = round(count_0/(count_0+count_1)*100,2)
# count_1_perc = round(count_1/(count_0+count_1)*100,2)

# print('Los ratios de desequilibrio en porcentaje con respecto a los datos de los pagadores y los morosos son: %.2f and %.2f'%(count_0_perc,count_1_perc))
# print('Ratios de desequilibrio es relativo con respecto a los datos de los pagadores y de los Morosos es %.2f : 1 (approx)'%(count_0/count_1))


# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots: 
# 1. Count plot of categorical column w.r.t TARGET; 
# 2. Percentage of defaulters within column

def univariate_categorical(feature,ylog=False,label_rotation=False,horizontal_layout=True):
    temp = applicationDF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=applicationDF,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
    
    # Define common styling
    ax1.set_title(feature, fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'}) 
    ax1.legend(['Repayer','Defaulter'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of Defaulters [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.show();



# function for plotting repetitive countplots in bivariate categorical analysis

def bivariate_bar(x,y,df,hue,figsize):
    
    plt.figure(figsize=figsize)
    sns.barplot(x=x,
                  y=y,
                  data=df, 
                  hue=hue, 
                  palette =['g','r'])     
        
    # Defining aesthetics of Labels and Title of the plot using style dictionaries
    plt.xlabel(x,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.ylabel(y,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.title(col, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.xticks(rotation=90, ha='right')
    plt.legend(labels = ['Repayer','Defaulter'])
    plt.show()
    
    
# function for plotting repetitive rel plots in bivaritae numerical analysis on applicationDF

def bivariate_rel(x,y,data, hue, kind, palette, legend,figsize):
    
    plt.figure(figsize=figsize)
    sns.relplot(x=x, 
                y=y, 
                data=applicationDF, 
                hue="TARGET",
                kind=kind,
                palette = ['g','r'],
                legend = False)
    plt.legend(['Repayer','Defaulter'])
    plt.xticks(rotation=90, ha='right')
    plt.show()


#function for plotting repetitive countplots in univariate categorical analysis on the merged df

def univariate_merged(col,df,hue,palette,ylog,figsize):
    plt.figure(figsize=figsize)
    ax=sns.countplot(x=col, 
                  data=df,
                  hue= hue,
                  palette= palette,
                  order=df[col].value_counts().index)
    

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})     
    else:
        plt.ylabel("Count",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})       

    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


def test_chi_square(applicationDF, column):
    """
    Chi cuadrado Genero buenos pagadoreso malos pagadores
    loan_process_df['TARGET']==0] # Repayers
    loan_process_df['TARGET']==1] # Defaulters
    Hipotesis: ¿El género del clientes está relacionada con el estatus de ser buen pagador, o mal pagador?
    """

    """
    Contigencia 
    """ 
    contigency= pd.crosstab(applicationDF[column], applicationDF['TARGET'])
    print('Tabla de contigencia')
    print(contigency)


    """
    Contigencia porcentajes
    """ 
    contigency_pct = pd.crosstab(applicationDF[column], applicationDF['TARGET'], normalize='all')
    print('Tabla de contigencia')
    print(contigency_pct)


    # Chi-square test of independence.
    c, p, dof, expected = chi2_contingency(contigency_pct)
    print('Giskard: ', p)

    """
    Conclusión: Debido a que el valor p = 0.99 > 0.05, aceptamos la hipótesis nula y creemos que no 
    hay una diferencia significativa en el genero del cliente/empresa solicitanate, con el estado de ser buen pagado o mal pagador.
    """

    plt.figure(figsize=(12,8))
    sns.heatmap(contigency_pct, annot=True, cmap="YlGnBu")
    plt.show()



"""
Resultados de segmentación
"""

# univariate_categorical('NAME_CONTRACT_TYPE',True)
# univariate_categorical('CODE_GENDER')
# univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)
# univariate_categorical("AGE_GROUP",False,False,True)
   
   
# #Comprobación del estado del contrato en función del estado de reembolso del préstamo y de si existe alguna pérdida comercial o financiera
# loan_process_df = pd.merge(applicationDF, previousDF, how='inner', on='SK_ID_CURR')
# loan_process_df.head()
# univariate_merged("NAME_CONTRACT_STATUS",loan_process_df,"TARGET",['g','r'],False,(12,8))
# g = loan_process_df.groupby("NAME_CONTRACT_STATUS")["TARGET"]
# df1 = pd.concat([g.value_counts(),round(g.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))
# df1['Percentage'] = df1['Percentage'].astype(str) +"%" # añadiendo el símbolo del porcentaje en los resultados para su comprensión
# print (df1)

# test_chi_square(applicationDF, "NAME_CONTRACT_TYPE")
# test_chi_square(applicationDF, "CODE_GENDER")
# test_chi_square(applicationDF, "NAME_EDUCATION_TYPE")
# test_chi_square(applicationDF, "AGE_GROUP")

"""
Métodos predictivos

"""
# Cambiamos la variable categórica CODE_GENDER por la variable numérica CODE_GENDER (Donde male = 1, fremale = 0)
applicationDF = pd.get_dummies(applicationDF, columns=['CODE_GENDER'], drop_first=True)
# print(categorize_data.head())
# # Seleccionamos las características para el modelo
data = applicationDF[['NAME_CONTRACT_TYPE_', 'GENDER_','AGE',"NAME_EDUCATION_TYPE_", "CNT_CHILDREN" , 'NAME_FAMILY_STATUS_','TARGET']]
# data = applicationDF[['NAME_CONTRACT_TYPE_', 'GENDER_','AGE',"NAME_EDUCATION_TYPE_",'TARGET']]
# print(data.head())
# X son nuestras variables independientes
X_independent = data.drop(["TARGET"],axis = 1)
# y es nuestra variable dependiente
y_dependet = data.TARGET
# # División 75% de datos para entrenamiento, 25% de daatos para test
# X_train, X_test, y_train, y_test = train_test_split(X_independent, y_dependet,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_independent, y_dependet,test_size=0.25, random_state=0)
# # Creaamos el modelo de Bosques Aleatorios (y configuramos el número de estimadores (árboles de decisión))
BA_model = RandomForestClassifier(n_estimators = 30, 
                                  random_state = 0,
                                  min_samples_leaf = 100,)
# """
# Entrenamiento 
# """
BA_model.fit(X_train, y_train)
# # Accuracy promedio
acurrancy_meave= BA_model.score(X_test, y_test)
print('RandomForestClassifier: ', acurrancy_meave)

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
acurrancy_meave=model.score(X_train, y_train)
print('GradientBoostingRegressor: ', acurrancy_meave)

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
target_column = ['TARGET'] 
predictors = list(set(list(data.columns))-set(target_column))
data[predictors] = data[predictors]/data[predictors].max()
X_independent = data[predictors].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X_independent, y, test_size=0.30, random_state=40)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
time_process = round((time.time() - start_time),2)
print('Giskard: Tiempo que se demoro en ejecutar', time_process, "Segundos")