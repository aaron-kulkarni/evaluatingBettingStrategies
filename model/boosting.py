import pandas as pd
import os
import sys
sys.path.append("/Users/shaolin/iCloud/Project/py_lib/")
import read_excel as myread
import strat 
import glob
import ray
import pdb
import time
import numpy_financial as npf
import datetime
import numpy as np
import ray

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

#https://www.anyscale.com/blog/how-to-distribute-hyperparameter-tuning-using-ray-tune

@ray.remote
def read_all(fl_):
    import sys
    import pandas as pd
    sys.path.append("/Users/shaolin/iCloud/Project/py_lib/")
    import read_excel as myread
    
    df_=pd.read_csv(fl_,header=[0,1],index_col=0)
    f_=fl_.split('/')[::-1][0]
    print(f_)
    #df_['_file']=f_
    #myread.clean_df_col(df_)
    print(df_.columns)
    return(df_)

@ray.remote
def read_all_elo(fl_):
    import sys
    import pandas as pd
    sys.path.append("/Users/shaolin/iCloud/Project/py_lib/")
    import read_excel as myread
    
    df_=pd.read_csv(fl_,index_col=0)
    f_=fl_.split('/')[::-1][0]
    print(f_)
    #df_['_file']=f_
    #myread.clean_df_col(df_)
    print(df_.columns)
    return(df_)


ray.init(num_cpus=3)
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/bettingOddsData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'adj' in i]
start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_odds=pd.concat(rslt)
df_odds_d = df_odds['homeProbAdj'] - df_odds['awayProbAdj']


fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/gameStats/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'game_state' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_game=pd.concat(rslt)


fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/teamStats/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'team' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_team=pd.concat(rslt)
# df_team stat include current game; cannot be used

fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'team' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_avg_team=pd.concat(rslt)
df_avg_team = df_avg_team.select_dtypes(include=np.number)

df_avg_team_d = df_avg_team['team'] / df_avg_team['opp']

col_df=df_avg_team.columns.get_level_values(1)[1:38]
fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/averageTeamData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'away' in i]

start_time = time.time()
rslt= ray.get([read_all.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_avg_team_away=pd.concat(rslt)
df_avg_team_away = df_avg_team.select_dtypes(include=np.number)


d = dict(zip(df_avg_team_away.columns.levels[0], ["away_opps","away_team"]))

df_avg_team_away  = df_avg_team_away.rename(columns=d, level=0)
df_avg_team_away_d = df_avg_team_away['away_team'] / df_avg_team_away['away_opps']

df_avg_team_away_std = pd.concat([df_avg_team_d,df_avg_team_away_d],keys=['home','away'],axis=1)
#df_avg_team_1=df_avg_team['team'][col_df]-df_avg_team['opp'][col_df]

fl_dir= '/Users/shaolin/iCloud/Project/modelEst/evaluatingBettingStrategies/data/eloData/'
fl_list=glob.glob(fl_dir+'*')

read_list=[i for i in fl_list if 'elo' in i]

start_time = time.time()
rslt= ray.get([read_all_elo.remote(f) for  f in read_list])
duration = time.time() - start_time
ray.shutdown() 
df_elo=pd.concat(rslt)
#df_elo = df_avg_team.select_dtypes(include=np.number)
cols = pd.MultiIndex.from_tuples([("Before", "homeTeamElo"), 
                                  ("Before", "awayTeamElo"), 
                                  ("After", "homeTeamElo"),
                                  ("After", "awayTeamElo")])
df_elo0=pd.DataFrame(columns=cols)
df_elo0[df_elo0.columns] = df_elo
df_elo_d=pd.DataFrame(columns=["elo_rt"])
df_elo_d['elo_rt']=df_elo['homeTeamElo']/df_elo['awayTeamElo']

col_df=df_avg_team.columns.get_level_values(1)[1:38]

#df_avg_team_1=df_avg_team['team'][col_df]-df_avg_team['opp'][col_df]

df_all = pd.concat([df_game,df_odds,df_team,df_avg_team,df_avg_team_away,df_elo0],axis=1)

df_all[('status','status')] = df_all.apply(lambda d: 1 if d['gameState']['winner']==d['gameState']['teamHome'] else 0,axis=1)

df_game_d =df_game['gameState'][['winner','teamHome','timeOfDay']]
df_game_d['avgSalary_rt'] = df_game['home']['avgSalary'] / df_game['away']['avgSalary'] 

df_all_d = pd.concat([df_game_d,df_odds_d,df_avg_team_away_d,df_elo_d],axis=1)
df_all_d['status'] = df_all_d.apply(lambda d: 1 if d['winner'] == d['teamHome'] else 0,axis=1)

game_col= [i for i in list(df_game.columns) if i[1] in ['streak','daysSinceLastGame','record','matchupWins','rivalry','avgSalary']]

game_col2=[i for i in list(df_avg_team) if  i[1] in ['Drtg','Ortg','eFG%','TS%']]

select_x=[i for i in list(df_all.columns) if 'homeProbAdj' in i  or i  in game_col or i in game_col2]
select_x=[i for i in list(df_all.columns) if 'homeProbAdj' in i  or i  in game_col or i in list(df_avg_team.columns) or i in list(df_avg_team_away.columns) or i[0]=='Before']
#select_x=[i for i in list(df_all.columns) if i  in game_col or i in list(df_avg_team) or i[0]=='Before']

XX=X.copy()
XX['oddwin']=X.apply(lambda d: 1 if d['homeProbAdj_Pinnacle (%)']>0.5 else 0 ,axis=1)
XX=pd.concat([y,XX],axis=1)

X=df_all[select_x]
y=df_all['status']
X=df_all_d[df_all_d.columns[3:60]]
y=df_all_d['status']


cat= [game_col[2],game_col[6],game_col[8]]
X[cat] = X[cat].astype('category')
X=X.drop(columns=cat)
X.columns = X.columns.get_level_values(0) + '_' +  X.columns.get_level_values(1)
print(X)
X['Drtg_d'] = X['team_Drtg'] - X['opp_Drtg']
X['Ortg_d'] = X['team_Ortg'] - X['opp_Ortg']


from sklearn.model_selection import train_test_split, cross_val_score
#from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.80, stratify=y, random_state=42)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, Y_train)
predictions = gb_clf2.predict(X_val)
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, Y_train)
score = xgb_clf.score(X_test, Y_test)
print(score)
from sklearn.model_selection import KFold, GridSearchCV
def xgb_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'max_depth': [4,5,6] , 'min_child_weight':[4,5,6] ,'learning_rate': [0.05,0.1,0.5] ,'n_estimators': [20,50,100] }
    # decision tree model
    xgb_model=XGBClassifier()
    #use gridsearch to test all values
    xgb_gscv = GridSearchCV(xgb_model, param_grid, cv=nfolds)
    #fit model to data
    xgb_gscv.fit(X, y)
    print(xgb_gscv.best_params_)
    print(xgb_gscv.best_estimator_)
    print(xgb_gscv.best_score_)

xgb_grid_search(X_train,Y_train,5)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))


print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")

dmat_train = xgb.DMatrix(X_train, Y_train, feature_names=X.columns, enable_categorical=True)
dmat_test = xgb.DMatrix(X_test, Y_test, feature_names=X.columns, enable_categorical=True)

#param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
#num_round = 2
#bst = xgb.train(param, dmat_train, num_round)

booster = xgb.train({'max_depth':4, 'eta': 0.02,'objective':'binary:logistic','max_cat_to_onehot': 5},dmat_train,num_boost_round=50,evals=[(dmat_train, "train"), (dmat_test, "test")])




print("\nTrain RMSE : ",booster.eval(dmat_train))
print("Test  RMSE : ",booster.eval(dmat_test))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_preds = [1 if pred>0.5 else 0 for pred in booster.predict(data=dmat_train)]
test_preds = [1 if pred>0.5 else 0 for pred in booster.predict(data=dmat_test)]

print("\nTest  Accuracy : %.3f"%accuracy_score(Y_test, test_preds))
print("Train Accuracy : %.3f"%accuracy_score(Y_train, train_preds))

dd=pd.concat([X_test,Y_test],axis=1)
dd['pred'] = booster.predict(data=dmat_test)

dd['pred_bkt']=pd.qcut(dd['pred'],[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
dd['odd_bkt']=pd.qcut(dd['homeProbAdj_Marathonbet (%)'],[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])


with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    xgb.plotting.plot_importance(booster, ax=ax, height=0.6, importance_type="weight")


plt.show()

with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    xgb.plotting.plot_tree(booster, ax=ax, num_trees=3)

plt.show()



booster = xgb.train({"tree_method": "hist", "max_cat_to_onehot": 5}, dmat_train)
# Must use JSON for serialization, otherwise the information is lost
#booster.save_model("categorical-model.json")
#SHAP value computation:
SHAP = booster.predict(dmat_train, pred_interactions=True)

# categorical features are listed as "c"
print(booster.feature_types)


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
pipe = Pipeline([
  #('preproc', PreprocTransformer(categorical_features, numerical_features)),
  ('fs', SelectKBest()),
  ('clf', xgb.XGBClassifier(objective='binary:logistic', enable_categorical=True, use_label_encoder=False))
])
search_space = [
  {
    'clf__n_estimators': [50, 100, 150, 200],
    'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'clf__max_depth': range(3, 10),
    'clf__colsample_bytree': [i/10.0 for i in range(1, 3)],
    'clf__gamma': [i/10.0 for i in range(3)],
    'fs__score_func': [chi2],
    'fs__k': [10],
  }
]

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# AUC and accuracy as score
scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score)}
# Define grid search
grid = GridSearchCV(
  pipe,
  param_grid=search_space,
  cv=kfold,
  scoring=scoring,
  refit='AUC',
  verbose=1,
  n_jobs=-1
)
# Fit grid search
model = grid.fit(X_train,Y_train)


res=xgb.cv(param, dmat_train, 10, nfold=5,
       metrics={'error'}, seed=0,
       callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)])

print('running cross validation, disable standard deviation display')

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
bst = xgb.train()

grid_search = GridSearchCV(model, param, scoring="neg_log_loss", n_jobs=-1, cv=kfold)


model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_full, y_train)

booster = xgb.train({'max_depth': 3, 'eta': 1, 'objective': 'reg:squarederror'},
                    dmat_train,
                    evals=[(dmat_train, "train"), (dmat_test, "test")])

y_pred_valid = model.predict(X_valid_full)

accuracy = accuracy_score(y_valid, y_pred_valid)
accuracy

y_pred_train = model.predict(X_train_full)

accuracy = accuracy_score(y_train, y_pred_train)
accuracy







import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

class StringIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace(
            {-1: len(s.cat.categories)}
        ))
    
transformer = Pipeline([
    ('features', FeatureUnion(n_jobs=1, transformer_list=[
        # Part 1
        ('boolean', Pipeline([
            ('selector', TypeSelector('bool')),
        ])),  # booleans close
        
        ('numericals', Pipeline([
            ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler()),
        ])),  # numericals close
        
        # Part 2
        ('categoricals', Pipeline([
            ('selector', TypeSelector('category')),
            ('labeler', StringIndexer()),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]))  # categoricals close
    ])),  # features close
])  # pipeline close

df = pd.DataFrame({
    'boolean_column': [True,False,True,False], 
    'integer_column': [1,2,3,4],
    'float_column': [1.,2.,3.,4.]
})
# Selecting booleans
boolean_columns = df.select_dtypes(include=['bool'])
# Selecting numericals
numerical_columns = df.select_dtypes(include=[np.number])

from catboost import Pool, CatBoostRegressor



cat_features = [5,6]
# initialize Pool
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, cat_features=cat_features)






from sklearn.pipeline import Pipeline

class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self

df = pd.DataFrame({
    'name':['alice','bob','charlie','david','edward'],
    'age':[24,32,np.nan,38,20]
})

# create a pipeline with a single transformer
pipe = Pipeline([
    ('selector', SelectColumnsTransformer([('status','status')]))
])

pipe.fit_transform(df_all)

X,y = 


pd.set_option('display.max_columns', None)

import pandas as pd
from sklearn.pipeline import Pipeline

class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

# this function takes a dataframe as input and
# returns a modified version thereof
def process_dataframe(input_df):
    input_df["text"] = input_df["text"].map(lambda t: t.upper())
    return input_df

# sample dataframe
df = pd.DataFrame({
    "id":[1,2,3,4],
    "text":["foo","Bar","BAz","quux"]
})

# this pipeline has a single step
pipeline = Pipeline([
    ("uppercase", DataframeFunctionTransformer(process_dataframe))
])

# apply the pipeline to the input dataframe


pipeline.fit_transform(df)
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

class ToDenseTransformer():

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self

# need to make matrices dense because PCA does not work with sparse vectors.
pipeline = Pipeline([
    ('to_dense',ToDenseTransformer()),
    ('pca',PCA()),
    ('clf',DecisionTreeClassifier())
])

pipeline.fit(sparse_data_matrix,target)
pipeline.predict(sparse_data_matrix)

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self

df = pd.DataFrame({
    'name':['alice','bob','charlie','david','edward'],
    'age':[24,32,np.nan,38,20]
})

pipe = Pipeline([
    ('selector', SelectColumnsTransformer(["name"]))
])

pipe.fit_transform(df)

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    'name':['alice','bob','charlie','david','edward'],
    'age':[24,32,np.nan,38,20]
})

transformer_step = ColumnTransformer([
        ('impute_mean', SimpleImputer(strategy='mean'), ['age'])
    ], remainder='passthrough')

pipe = Pipeline([
    ('select', select)
])

# fit as you would a normal transformer
pipe.fit(features)

# transform the input
pipe.transform(features)
















import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
drive = GoogleDrive(gauth)



from google.colab import drive

import requests
from io import StringIO

orig_url='https://drive.google.com/file/d/1KDubPBEU4RQzBaORAFORIpCmMGXxvNFE/view?usp=sharing'
orig_url='https://drive.google.com/file/d/1zZr6q3zYmJBVnejFyvsS-LXEe3Xj3S7P/view?usp=sharing'
file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
dfs = pd.read_csv(dwn_url)



import requests


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

    



import pandas as pd
response = {
 "kind": "drive#childList",
 "etag": "\"9NuiSicPg_3yRScMQO3pipPxwvs\"",
 "selfLink": "https://www.googleapis.com/drive/v2/files/1IkO_nB83mUfKLopEtYsNT7RbMOIcWDAK/children",
 "items": [
  {
   "kind": "drive#childReference",
   "id": "1YtG84A9ZJNM7A3OgD3nOQk8V9bz_mYQ1",
   "selfLink": "https://www.googleapis.com/drive/v2/files/1IkO_nB83mUfKLopEtYsNT7RbMOIcWDAK/children/1YtG84A9ZJNM7A3OgD3nOQk8V9bz_mYQ1",
   "childLink": "https://www.googleapis.com/drive/v2/files/1YtG84A9ZJNM7A3OgD3nOQk8V9bz_mYQ1"
  },
  {
   "kind": "drive#childReference",
   "id": "14P3NAdGid-iJl1JSFDHLb8U3BRb0tOQO",
   "selfLink": "https://www.googleapis.com/drive/v2/files/1IkO_nB83mUfKLopEtYsNT7RbMOIcWDAK/children/14P3NAdGid-iJl1JSFDHLb8U3BRb0tOQO",
   "childLink": "https://www.googleapis.com/drive/v2/files/14P3NAdGid-iJl1JSFDHLb8U3BRb0tOQO"
  }
 ]
}

item_arr = []
for item in response["items"]:
    print(item["id"])
    download_url = 'https://drive.google.com/uc?id=' + item["id"]
    item_arr.append(pd.read_csv(download_url))
df = pd.concat(item_arr, axis=0)
print(df.head())

URL='https://drive.google.com/drive/folders/1c9rePU207eaJmsoNzYx3hzrgkE5wO3eD?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
#df = pd.read_pickle(path)
df = pd.read_csv(path)
