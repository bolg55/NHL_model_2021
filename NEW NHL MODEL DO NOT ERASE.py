#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sqlalchemy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


# In[2]:


# SQL CONNECTION

engine = sqlalchemy.create_engine('mysql+pymysql://root:Sp1d3rman@localhost:3306/new_nhl_data')


# In[3]:


# NEW NHL MODEL

# Reading in all games data from nhl_data SQL db.
df_sql = pd.read_sql("game_data", engine)

#Checking that last nights games were added correctly.

df_sql.tail()


# In[4]:


df_sql.reset_index().set_index(['index','Date']).head()


# In[5]:


df_sql = df_sql.replace([np.inf,-np.inf,'-'],np.nan)


# In[6]:


predictors = [x for x in df_sql.columns if x not in ['Team','Teamopp','Date','W','L']]


# In[7]:


df_sql[predictors] = df_sql[predictors].astype(float)


# In[8]:


print("printing row index with infinity ") 
r = df_sql.index[np.isinf(df_sql[predictors]).any(1)] 
print(r)


# ## Convert into rolling measures

# In[9]:


rolling_df = df_sql.reset_index().set_index(['index','Date','W','L']).groupby('Team').rolling(10,min_periods=10).mean().shift(1)


# In[10]:


rolling_df.head()


# In[11]:


final = rolling_df.reset_index(level=['Team','Date','W','L']).sort_index()


# In[12]:


final.head()


# In[13]:


df_a = final.loc[df_sql.index % 2 == 0, :]
df_b = final.loc[df_sql.index % 2 != 0, :]


# In[14]:


df_c = pd.concat([df_a,df_b.set_index(df_a.index)], axis =1)


# In[15]:


suffix = 'opp'
df_c.columns = [name if duplicated == False else name + suffix for duplicated, name in zip(df_c.columns.duplicated(), df_c.columns)]


# In[16]:


df_c.set_index('Date', inplace = True)


# In[17]:


df = df_c.drop(columns=['Team','Teamopp','Dateopp','GP','TOI/GP','L','OTL','ROW','Points','Point %','GPopp','TOI/GPopp','Wopp','Lopp','OTLopp','ROWopp','Pointsopp','Point %opp','idopp'])
df.head()


# In[18]:


check_null = df.isnull().sum()


# In[19]:


check_null[check_null.gt(5000)]


# In[20]:


df.drop(['HDGF%','MDGF%','LDGF%','HDGF%opp','MDGF%opp','LDGF%opp'],inplace=True,axis=1)


# In[21]:


# df.shape


# In[22]:


#pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.float_format', lambda x: '%.3f' % x,'display.width', 320)


# In[23]:


predictors = [x for x in df.columns if x not in ['Team','Teamopp','Date','W','L']]


# In[24]:


df = df.dropna()


# In[25]:


X = df.drop(columns=['W'])
y = df["W"]

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
model = LogisticRegression(class_weight= 'balanced', max_iter= 10000)
model.fit(X_train,y_train)


# In[27]:


model.score(X_test, y_test)


# ## Fetching live data and predicting on it

# In[28]:


games = pd.read_html('https://www.hockey-reference.com/leagues/NHL_2021_games.html')
games[0].drop(columns = ['Unnamed: 5','G','G.1','Att.','LOG','Notes'], inplace = True)
games = games[0]
games['Date'] = pd.to_datetime(games["Date"],format="%Y-%m-%d")
today = pd.Timestamp("today").floor("D")
games = games.loc[(games.Date == today)]


# In[29]:


games


# In[30]:


# Initialise columns for each predictor column
for col in predictors:
    games[col] = np.nan


# In[31]:


def update_row_with_features(row):

    # Fetch the last 10 games mean from original dataset for the particular teams of the game
    home_rec = df_sql[df_sql.Team.eq(row['Home'])].iloc[-10:].mean()
    visit_rec = df_sql[df_sql.Team.eq(row['Visitor'])].iloc[-10:].mean()
    home_rec.index = [x+'opp' for x in home_rec.index]

    #Convert into dictionary for easier addition to dataframe
    home_rec = home_rec.to_dict()
    visit_rec = visit_rec.to_dict()
    home_rec.update(visit_rec)
    
    #Update dataframe row using dictionary
    for k,v in home_rec.items():
        if k in predictors:
            games.loc[row.name,k] = v


# In[32]:


# Add feature values to each row of dataframe for predictions
games.apply(update_row_with_features,axis=1)


# In[33]:


games


# In[34]:


out = pd.DataFrame(data = {'v_team': games['Visitor'], 'v_prob': np.round(model.predict_proba(scaler.transform(games[predictors]))[:,1],3),'h_prob': np.round(model.predict_proba(scaler.transform(games[predictors]))[:,0],3),'h_team': games['Home']})


# In[35]:


out = pd.DataFrame(data = {'v_team': games['Visitor'], 'v_prob': np.round(model.predict_proba(scaler.transform(games[predictors]))[:,1],3),'v_odds': np.round(1 / out['v_prob'],2),'h_prob': np.round(model.predict_proba(scaler.transform(games[predictors]))[:,0],3),'h_odds':np.round(1 / out['h_prob'],2),'h_team': games['Home']})


# In[36]:


out


# In[37]:


out.to_csv('daily projections.csv')


# In[ ]:




