from dash import Dash, html, callback, Input, Output
from PIL import Image
import dash_bootstrap_components as dbc
import os
import numpy as np

import sys
sys.path.append('src')
# print(sys.path)
from shap_explainer import shap_explainer
from grad_cam_explainer import grad_cam

def show_explainer(vehicle, file_name):
    raw_image = Image.open(f"dash_src/assets/images/raw/{vehicle}/{file_name}")
    preprocessed_image = Image.open(f"dash_src/assets/images/preprocessed/{vehicle}/{file_name}")
    return dbc.Card(
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        children = [
                            html.Div(
                                children=[
                                    html.H1(vehicle, style={'text-align': 'center'}),
                                    html.Div([
                                        html.H3(f'Original Image', style={'text-align': 'center'}),
                                        html.Img(src=raw_image, style={'margin-left': 'auto', 'margin-right': 'auto', 'padding': '5px', 'display': 'block'}),
                                    ]),
                                    html.Div([
                                        html.H3(f'Preprocessed Image', style={'text-align': 'center'}),
                                        html.Img(src=preprocessed_image, style={'margin-left': 'auto', 'margin-right': 'auto', 'padding': '5px', 'display': 'block'})
                                    ]),
                                ],
                                
                            )
                        ]
                        
                    ),
                    dbc.Col(
                        children = [
                            create_explanation(vehicle, file_name)
                        ]
                    )
                ]
            )
        ]
    )

# @callback(
#     Output('start-explain-output', 'children'),
#     Input('start-explain-input', 'value')
# )
def create_explanation(vehicle, file_name):
    file_path = f'dash_src/assets/images/preprocessed/{vehicle}/{file_name}'
    image = Image.open(file_path)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    shap_image = shap_explainer(image, ['Bike', 'Car'], vehicle, file_name)
    grad_cam_image = grad_cam(image, vehicle, file_name)
    return html.Div(
        children=[
            html.Div(
                children= [
                html.H1('Explanation', style={'text-align': 'center'}),
                html.H2('SHAP', style={'text-align': 'center'}),
                html.Img(src=shap_image,
                        style={'height': '300px',
                                'width': '300px',
                                'margin': '5px',
                                'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block'}),
                ],
                style={'align-items': 'left'}
                ),
            html.Div(
                children=[
                    html.H2('Grad-CAM', style={'text-align': 'center'}),
                    html.Img(src=grad_cam_image,
                    style={'height': '300px',
                            'width': '250px',
                            'margin': '5px',
                            'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block'}),
                ],
                style={'align-items': 'left'}
            )
        ]
    )