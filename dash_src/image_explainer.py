from dash import Dash, html, callback, Input, Output
from PIL import Image
import dash_bootstrap_components as dbc
import os
import numpy as np

import sys
sys.path.append('src')
from shap_explainer import shap_Explainer
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
                            html.H1(vehicle),
                            html.Img(src=raw_image),
                            html.Img(src=preprocessed_image)
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
    shap_image = shap_Explainer(image, ['Bike', 'Car'], vehicle, file_name)
    grad_cam_image = grad_cam(image, vehicle, file_name)
    return html.Div(
        children=[
            html.H1('Explanation'),
            html.H2('SHAP'),
            html.Img(src=shap_image,
                     style={'height': '300px',
                            'width': '300px',
                            'margin': '5px'}),
            html.H2('Grad-CAM'),
            html.Img(src=grad_cam_image,
                     style={'height': '300px',
                            'width': '300px',
                            'margin': '5px'}),
        ]
    )