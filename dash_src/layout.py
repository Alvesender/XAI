import dash
from dash import Dash, html, callback, Input, Output, dcc, State
from PIL import Image
import dash_bootstrap_components as dbc
import os
from image_explainer import show_explainer
from display_all_vehicles import display_vehicles


def create_layout(app :Dash) -> html.Div:
    return html.Div(
        children=[
            html.Div(id='explainer'),
            dbc.Row(
                children=[
                    display_vehicles("Bike"),
                    display_vehicles("Car"),
                ],
            )
        ]
    )









