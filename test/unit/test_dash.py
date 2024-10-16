import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from dash_bootstrap_components import Table, themes


# Create a Pandas DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol'],
    'Age': [25, 30, 35]
})

# Create a Dash app
app = Dash(__name__, external_stylesheets=[themes.BOOTSTRAP])

# Create a Bootstrap table
table = Table.from_dataframe(df, striped=True, bordered=True, hover=True)

# Add the table to the app layout
app.layout = html.Div([table])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)