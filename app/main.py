#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import math
import lightgbm as lgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV



X_avg=['FE_RESIN_LOAD','TA04_L2_5_L_SIC0726_PV','TA04_L2_5_L_MIC0730_PV','TA04_L2_5_WIC0728_PV' ,'TA04_L2_5_GREC_DENS_PV','FE_ESPESOR_GRECON','FE_CALCULATED_DENSITY_MAT',
'FE_SAWDUST_RATIO', 'TA04_L2_4_H_WI0386_PV','FE_EXTERNAL_CHIP_RATIO', 'FE_PRESSURE_1'  ,'FE_PRESSURE_2_3',
'FE_PRESSURE_4_5','FE_PRESSURE_6_7','FE_PRESSURE_8_9','FE_PRESSURE_10_11','FE_PRESSURE_12_13','FE_PRESSURE_14_15',
'FE_PRESSURE_16_17','FE_THICKNESS_20','FE_THICKNESS_22','FE_THICKNESS_25','FE_THICKNESS_29','FE_THICKNESS_33','FE_THICKNESS_37']


# In[2]:



df = pd.read_csv('fulldata_201903_202006.csv', sep=',')
product = 'Ultralight 12'
print(df['ds'].min(), df['ds'].max())

df['ds'] = pd.to_datetime(df['ds'])
df['boarddate'] = pd.to_datetime(df['boarddate'])
df = df.sort_values(by=['ds'], ascending=True)

df_q = df.copy()
df_q = df_q[df_q['Product']== product]
y = 'IB_avg'
print(df_q['ds'].min(), df_q['ds'].max())


# In[3]:


train_start_p1 = '2019-03-24'
train_end_p1 = '2019-03-12'
train_start_p2 = '2019-04-01'
train_end_p2 = '2019-10-30'
test_start = '2020-02-01'
test_end = '2020-05-20'


# In[4]:


df_train_a = df_q[(df_q.ds >= train_start_p1) & (df_q.ds <= train_end_p1)].copy()
df_train_b = df_q[(df_q.ds >= train_start_p2) & (df_q.ds <= train_end_p2)].copy()
df_train = pd.concat([df_train_a, df_train_b], ignore_index=True)
df_test = df_q[(df_q.ds >= test_start) & (df_q.ds <= test_end)].copy()


# In[5]:


for x in X_avg :
    df_train[x] = df_train[x].astype(float)
    df_train[x].fillna(method='bfill',inplace=True)


# In[6]:




# model = lgb.LGBMRegressor(subsample= 0.6, reg_lambda= 1.0, reg_alpha= 50.0, n_estimators= 100, 
#                           min_child_weight= 10.0, max_depth= 6, learning_rate= 0.01, gamma=0.25,
#                           eval_metric='mae', colsample_bytree= 0.4, colsample_bynode= 0.4, colsample_bylevel= 0.4)

model = lgb.LGBMRegressor()
model.fit(df_train[X_avg], df_train[y])


df_feature_importances = pd.DataFrame(model.feature_importances_*100,columns=["Importance"],index=X_avg)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)


# In[7]:




fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)

slider_0_label =  'FE_RESIN_LOAD'
slider_0_min = round( df_test[slider_0_label].min(),2)
slider_0_mean = round(df_test[slider_0_label].mean(),2)
slider_0_max =  round(df_test[slider_0_label].max(),2)
range0=round(((slider_0_max-slider_0_min)/5),4)



slider_1_label = 'TA04_L2_5_L_SIC0726_PV'
slider_1_min =  round(df_test[slider_1_label].min(),2)
slider_1_mean = round(df_test[slider_1_label].mean(),2)
slider_1_max =  round(df_test[slider_1_label].max(),2)
range1=round(((slider_1_max-slider_1_min)/5),4)

slider_2_label ='TA04_L2_5_L_MIC0730_PV'
slider_2_min = round( df_test[slider_2_label].min(),2)
slider_2_mean = round(df_test[slider_2_label].mean(),2)
slider_2_max =  round(df_test[slider_2_label].max(),2)
range2=round(((slider_2_max-slider_2_min)/5),4)


# In[8]:


def seq_floats(start, stop, step=1):
    stop = stop - step;
    number = int(round((stop - start)/float(step)))

    if number > 1:
        return([start + step*i for i in range(number+1)])

    elif number == 1:
        return([start])

    else:
        return([])
    


# In[ ]:




app = dash.Dash()
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        
                        html.H1(children="Simulation TEST"),
                        
                        
                        dcc.Graph(figure=fig_features_importance),
                        
                        html.H4(children=slider_0_label),

                        
                        dcc.Slider(
                            id='X0_slider',
                           min=slider_0_min,
                           max=slider_0_max,
                           value=slider_0_mean,
                           step=100,
                           marks={opacity: f'{opacity:.1f}' for opacity in seq_floats(slider_0_min, slider_0_max, range0)}
                            ),
                        
                        
                        
                        html.H4(children=slider_1_label),

                        
                        dcc.Slider(
                            id='X1_slider',
                           min=slider_1_min,
                           max=slider_1_max,
                           value=slider_1_mean,
                           step=None,
                           marks={opacity: f'{opacity:.1f}' for opacity in seq_floats(slider_1_min, slider_1_max, range1)}
                            ),

                       
                        html.H4(children=slider_2_label),

                        dcc.Slider(
                           id='X2_slider',
                           min=slider_2_min,
                           max=slider_2_max,
                           value=slider_2_mean,
                           step=None,
                            marks={opacity: f'{opacity:.1f}' for opacity in seq_floats(slider_2_min, slider_2_max, range2)}
                        ),

                       
                        
                        
                         html.H2(id="prediction_result"),

                    ])


# In[ ]:



@app.callback(Output(component_id="prediction_result",component_property="children"),

              [Input("X0_slider","value"),Input("X1_slider","value"), Input("X2_slider","value")])






def update_prediction(X0,X1, X2):

    
    input_X = np.array([X0,X1,X2,df_test['TA04_L2_5_WIC0728_PV'].mean(),
df_test['TA04_L2_5_GREC_DENS_PV'].mean(),
df_test['FE_ESPESOR_GRECON'].mean(),
df_test['FE_CALCULATED_DENSITY_MAT'].mean(),
df_test['FE_SAWDUST_RATIO'].mean(),
df_test['TA04_L2_4_H_WI0386_PV'].mean(),
df_test['FE_EXTERNAL_CHIP_RATIO'].mean(),
df_test['FE_PRESSURE_1'].mean(),
df_test['FE_PRESSURE_2_3'].mean(),
df_test['FE_PRESSURE_4_5'].mean(),
df_test['FE_PRESSURE_6_7'].mean(),
df_test['FE_PRESSURE_8_9'].mean(),
df_test['FE_PRESSURE_10_11'].mean(),
df_test['FE_PRESSURE_12_13'].mean(),
df_test['FE_PRESSURE_14_15'].mean(),
df_test['FE_PRESSURE_16_17'].mean(),
df_test['FE_THICKNESS_20'].mean(),
df_test['FE_THICKNESS_22'].mean(),
df_test['FE_THICKNESS_25'].mean(),
df_test['FE_THICKNESS_29'].mean(),
df_test['FE_THICKNESS_33'].mean(),
df_test['FE_THICKNESS_37'].mean()]).reshape(1,-1)        
    
    
    prediction = model.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Predicción IB a (Ultraliviano-12) : {}".format(round(prediction,2))

if __name__ == "__main__":
    #app.run_server()
    app.run_server(host='0.0.0.0',debug=True, port=8080)
    


# In[ ]:




