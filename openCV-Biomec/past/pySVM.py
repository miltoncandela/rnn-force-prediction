## SVM

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.graph_objs as go

df = pd.read_csv('Data/featureBoth1.csv')
print(df)

df = df[['a-g_P8', 't-g_P7', 'Clas']]

X = df.drop('Clas', axis = 1).to_numpy()
y_text = df['Clas'].to_numpy()
#y = (2 * LabelEncoder().fit_transform(y_text)) - 1
y = LabelEncoder().fit_transform(y_text)

points_colorscale = [
                     [0.0, 'rgb(199, 204, 249)'],
                     [0.5, 'rgb(235, 185, 177)'],
                     [1.0, 'rgb(159, 204, 186)']
                    ]

points = go.Scatter(
                    x = df['a-g_P8'],
                    y = df['t-g_P7'],
                    mode ='markers',
                    marker = dict(color = y,
                                colorscale = points_colorscale)
                   )
layout = go.Layout(
                   xaxis = dict(range = [0, 2]),
                   yaxis = dict(range = [-4, 6])
                  )

fig = go.Figure(data = [points], layout = layout)
fig.show()

(X_train, X_vt, y_train, y_vt) = train_test_split(X, y, test_size = 0.3, random_state = 0)
(X_validation, X_test, y_validation, y_test) = train_test_split(X_vt, y_vt, test_size = 0.5, random_state = 0)

svm = SVC(kernel = 'linear')

svm.fit(X_train, y_train)

decision_colorscale = [
                     [0.0, 'rgb(199, 204, 249)'],
                     [0.5, 'rgb(235, 185, 177)'],
                     [1.0, 'rgb(159, 204, 186)']
                      ]

detail_steps = 100

#(x_vis_0_min, x_vis_1_min) = (-1.05, -1.05) #X_train.min(axis=0)
#(x_vis_0_max, x_vis_1_max) = ( 1.05,  1.05) #X_train.max(axis=0)

(x_vis_0_min, x_vis_1_min) = X_train.min(axis = 0)
(x_vis_0_max, x_vis_1_max) = X_train.max(axis = 0)

x_vis_0_range = np.linspace(x_vis_0_min, x_vis_0_max, detail_steps)
x_vis_1_range = np.linspace(x_vis_1_min, x_vis_1_max, detail_steps)

(XX_vis_0, XX_vis_1) = np.meshgrid(x_vis_0_range, x_vis_0_range)

X_vis = np.c_[XX_vis_0.reshape(-1), XX_vis_1.reshape(-1)]

YY_vis = svm.decision_function(X_vis).reshape(XX_vis_0.shape)

points = go.Scatter(
                    x=df['a-g_P8'],
                    y=df['t-g_P7'],
                    mode='markers',
                    marker=dict(
                                color=y,
                                colorscale=points_colorscale),
                    showlegend=False
                   )
SVs = svm.support_vectors_
support_vectors = go.Scatter(
                             x=SVs[:, 0],
                             y=SVs[:, 1],
                             mode='markers',
                             marker=dict(
                                         size=15,
                                         color='black',
                                         opacity = 0.1,
                                         colorscale=points_colorscale),
                             line=dict(dash='solid'),
                             showlegend=False
                            )

decision_surface = go.Contour(x=x_vis_0_range,
                              y=x_vis_1_range,
                              z=YY_vis,
                              contours_coloring='lines',
                              line_width=2,
                              contours=dict(
                                            start=0,
                                            end=0,
                                            size=1),
                              colorscale=decision_colorscale,
                              showscale=False
                             )

margins = go.Contour(x=x_vis_0_range,
                     y=x_vis_1_range,
                     z=YY_vis,
                     contours_coloring='lines',
                     line_width=2,
                     contours=dict(
                                   start=-1,
                                   end=1,
                                   size=2),
                     line=dict(dash='dash'),
                     colorscale=decision_colorscale,
                     showscale=False
                    )

fig2 = go.Figure(data=[margins, decision_surface, support_vectors, points], layout=layout)
fig2.show()

svm_p2 = SVC(kernel = 'poly', degree = 2)

svm_p2.fit(X_train, y_train)

YY_vis_p2 = svm_p2.decision_function(X_vis).reshape(XX_vis_0.shape)

SVs_p2 = svm_p2.support_vectors_
support_vectors_p2 = go.Scatter(
                                x=SVs_p2[:, 0],
                                y=SVs_p2[:, 1],
                                mode='markers',
                                marker=dict(
                                            size=15,
                                            color='black',
                                            opacity = 0.1,
                                            colorscale=points_colorscale),
                                line=dict(dash='solid'),
                                showlegend=False
                               )

decision_surface_p2 = go.Contour(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YY_vis_p2,
                                 contours_coloring='lines',
                                 line_width=2,
                                 contours=dict(
                                               start=0,
                                               end=0,
                                               size=1),
                                 colorscale=decision_colorscale,
                                 showscale=False
                                )

margins_p2 = go.Contour(x=x_vis_0_range,
                        y=x_vis_1_range,
                        z=YY_vis_p2,
                        contours_coloring='lines',
                        line_width=2,
                        contours=dict(
                                      start=-1,
                                      end=1,
                                      size=2),
                        line=dict(dash='dash'),
                        colorscale=decision_colorscale,
                        showscale=False
                       )

fig3 = go.Figure(data=[margins_p2, decision_surface_p2, support_vectors_p2, points], layout=layout)
fig3.show()

svm_p3 = SVC(kernel = 'poly', degree = 3)
svm_p3.fit(X_train, y_train)

YY_vis_p3 = svm_p3.decision_function(X_vis).reshape(XX_vis_0.shape)

SVs_p3 = svm_p3.support_vectors_
support_vectors_p3 = go.Scatter(
                                x=SVs_p3[:, 0],
                                y=SVs_p3[:, 1],
                                mode='markers',
                                marker=dict(
                                            size=15,
                                            color='black',
                                            opacity = 0.1,
                                            colorscale=points_colorscale),
                                line=dict(dash='solid'),
                                showlegend=False
                               )

decision_surface_p3 = go.Contour(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YY_vis_p3,
                                 contours_coloring='lines',
                                 line_width=2,
                                 contours=dict(
                                               start=0,
                                               end=0,
                                               size=1),
                                 colorscale=decision_colorscale,
                                 showscale=False
                                )

margins_p3 = go.Contour(x=x_vis_0_range,
                        y=x_vis_1_range,
                        z=YY_vis_p3,
                        contours_coloring='lines',
                        line_width=2,
                        contours=dict(
                                      start=-1,
                                      end=1,
                                      size=2),
                        line=dict(dash='dash'),
                        colorscale=decision_colorscale,
                        showscale=False
                       )

fig4 = go.Figure(data=[margins_p3, decision_surface_p3, support_vectors_p3, points], layout=layout)
fig4.show()

svm_r = SVC(kernel = 'rbf')

svm_r.fit(X_train, y_train)

YY_vis_r = svm_r.decision_function(X_vis).reshape(XX_vis_0.shape)

SVs_r = svm_r.support_vectors_
support_vectors_r = go.Scatter(
                                x=SVs_r[:, 0],
                                y=SVs_r[:, 1],
                                mode='markers',
                                marker=dict(
                                            size=15,
                                            color='black',
                                            opacity = 0.1,
                                            colorscale=points_colorscale),
                                line=dict(dash='solid'),
                                showlegend=False
                               )

decision_surface_r = go.Contour(x=x_vis_0_range,
                                 y=x_vis_1_range,
                                 z=YY_vis_r,
                                 contours_coloring='lines',
                                 line_width=2,
                                 contours=dict(
                                               start=0,
                                               end=0,
                                               size=1),
                                 colorscale=decision_colorscale,
                                 showscale=False
                                )

margins_r = go.Contour(x=x_vis_0_range,
                        y=x_vis_1_range,
                        z=YY_vis_r,
                        contours_coloring='lines',
                        line_width=2,
                        contours=dict(
                                      start=-1,
                                      end=1,
                                      size=2),
                        line=dict(dash='dash'),
                        colorscale=decision_colorscale,
                        showscale=False
                       )

fig5 = go.Figure(data=[margins_r, decision_surface_r, support_vectors_r, points], layout=layout)
fig5.show()

yhat_train = svm.predict(X_train)
yhat_validation = svm.predict(X_validation)

print(accuracy_score(yhat_train, y_train), accuracy_score(yhat_validation, y_validation))

yhat_train_p2 = svm_p2.predict(X_train)
yhat_validation_p2 = svm_p2.predict(X_validation)

print(accuracy_score(yhat_train_p2, y_train), accuracy_score(yhat_validation_p2, y_validation))

yhat_train_p3 = svm_p3.predict(X_train)
yhat_validation_p3 = svm_p3.predict(X_validation)

print(accuracy_score(yhat_train_p3, y_train), accuracy_score(yhat_validation_p3, y_validation))

yhat_train_r = svm_r.predict(X_train)
yhat_validation_r = svm_r.predict(X_validation)

print(accuracy_score(yhat_train_r, y_train), accuracy_score(yhat_validation_r, y_validation))

