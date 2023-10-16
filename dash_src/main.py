from dash import Dash, html
import os
from layout import create_layout
import dash_bootstrap_components as dbc


app = Dash(__name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
app.title = "Car vs Bike"

app.layout = create_layout(app)

if __name__ == "__main__":
    app.run_server(debug=True)
