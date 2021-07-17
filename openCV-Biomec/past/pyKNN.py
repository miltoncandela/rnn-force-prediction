## KNN

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('Data/featureBoth1.csv')
df['Clas'] = df['Clas'].astype(object)
print(df)

df = df[['a-g_P8', 't-g_P7', 'Clas']]

fig = px.scatter(df, x = "a-g_P8", y = "t-g_P7", color = "Clas")
fig.show()

X = df.drop('Clas', axis = 1).to_numpy()
y_text = df['Clas'].to_numpy()
y = LabelEncoder().fit_transform(y_text)

(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size = 0.3, random_state = 0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size = 0.5, random_state = 0)

knn = KNeighborsClassifier(p = 2)

knn.fit(X_train, y_train)

detail_steps = 500

(x_vis_0_min, x_vis_0_max) = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
(x_vis_1_min, x_vis_1_max) = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

x_vis_0_range = np.linspace(x_vis_0_min, x_vis_0_max, detail_steps)
x_vis_1_range = np.linspace(x_vis_1_min, x_vis_1_max, detail_steps)

(XX_vis_0, XX_vis_1) = np.meshgrid(x_vis_0_range, x_vis_1_range)
X_vis = np.c_[XX_vis_0.reshape(-1), XX_vis_1.reshape(-1)]

yhat_vis = knn.predict(X_vis)
YYhat_vis = yhat_vis.reshape(XX_vis_0.shape)

region_colorscale = [
                     [0.0, 'rgb(199, 204, 249)'],
                     [0.5, 'rgb(235, 185, 177)'],
                     [1.0, 'rgb(159, 204, 186)']
                    ]
points_colorscale = [
                     [0.0, 'rgb(99, 110, 250)'],
                     [0.5, 'rgb(239, 85, 59)'],
                     [1.0, 'rgb(66, 204, 150)']
                    ]
fig2 = go.Figure(
                data=[
                      go.Heatmap(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YYhat_vis,
                                 colorscale=region_colorscale,
                                 showscale=False),
                      go.Scatter(x=df['a-g_P8'], 
                                 y=df['t-g_P7'],
                                 mode='markers',
                                 text=df['Clas'],
                                 name='',
#                                 showscale=False,
                                 marker=dict(
                                             color=y,
                                             colorscale=points_colorscale
                                            )
                                )
                     ],
                     layout=go.Layout(
                                      xaxis=dict(title='a-g_P8'),
                                      yaxis=dict(title='t-g_P7')
                                     )
               )
fig2.show()

yhat_train = knn.predict(X_train)
accuracy_score(yhat_train, y_train)

yhat_validation = knn.predict(X_validation)
accuracy_score(yhat_validation, y_validation)

knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train,y_train)

yhat3_vis = knn3.predict(X_vis)
YYhat3_vis = yhat3_vis.reshape(XX_vis_0.shape)

fig3 = fig2
fig3['data'][0]['z'] = YYhat3_vis

fig3.show()

yhat_train3 = knn3.predict(X_train)
accuracy_score(yhat_train3, y_train)

yhat_validation3 = knn3.predict(X_validation)
accuracy_score(yhat_validation3, y_validation)

knn5 = KNeighborsClassifier(n_neighbors = 5)
knn5.fit(X_train, y_train)

yhat5_vis = knn5.predict(X_vis)
YYhat5_vis = yhat3_vis.reshape(XX_vis_0.shape)

fig4 = fig2
fig4['data'][0]['z'] = YYhat5_vis

fig4.show()

yhat_train5 = knn5.predict(X_train)
accuracy_score(yhat_train5, y_train)

yhat_validation5 = knn5.predict(X_validation)
accuracy_score(yhat_validation5, y_validation)

