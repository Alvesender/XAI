import os
import dash
from dash import Dash, html, callback, Input, Output, dcc, State
from PIL import Image
import dash_bootstrap_components as dbc
from image_explainer import show_explainer
import pandas as pd

number_of_images = 100

def display_vehicles(vehicle :str)-> html.Div:
    # images = [Image.open(f"input/preprocessed/{vehicle}/{file}") for file in os.listdir(f"input/preprocessed/{vehicle}")[:100]]
    # image_divs = [dbc.Col([html.Img(src=image, alt=f'{vehicle}/{file}')]) for image in images]
    pred_data = {}
    pred_data['Bike'] = pd.read_csv('dash_src/assets/data/Bike_pred.csv')
    pred_data['Car'] = pd.read_csv('dash_src/assets/data/Car_pred.csv')
    
    image_divs =[]
    for i in range(number_of_images):
        image = Image.open(f'dash_src/assets/images/raw/{vehicle}/{vehicle}_{i}.png')
        pred_class = spectrum_to_vehicle(pred_data[vehicle].iloc[i]['pred_class'])
        image_divs.append(
            dbc.Col(
                dbc.Card(
                    children=[
                        html.Img(
                            src=image,
                            id=f'pic-{vehicle}_{i}',
                            style={'cursor': 'pointer',
                                    'height': '100px',
                                    'width': '100px',
                                    'margin': '5px'}
                            ),
                            'Predicted Class: ', 
                            html.B(pred_class, style={'color': color_the_vehicle_name(vehicle, pred_class)})
                        ]
                    )
                )
            )
    
    return dbc.Col(
        dbc.Card(
            children=[
                html.H1(vehicle),
                dbc.Row(
                    children=image_divs
                ),
            ],
        ),
    )
    
@callback(
    Output('explainer', 'children'),
    [Input(f'pic-Bike_{i}', 'n_clicks') for i in range(number_of_images)],
    [Input(f'pic-Car_{i}', 'n_clicks') for i in range(number_of_images)],
    [State(f'pic-Bike_{i}', 'alt') for i in range(number_of_images)],
    [State(f'pic-Car_{i}', 'alt') for i in range(number_of_images)],
    prevent_initial_call=True
)
def show_explanation(*args):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    vehicle = triggered_id.split('-')[1].split('_')[0]
    file_name = triggered_id.split('-')[1] + '.png'
    return show_explainer(vehicle, file_name)

def spectrum_to_vehicle(spectrum :float):
    if spectrum <= 0.5:
        return 'Bike'
    else:
        return 'Car'
    
def color_the_vehicle_name(true_vehicle, pred_vehicle):
    return 'green' if true_vehicle == pred_vehicle else 'red'
    