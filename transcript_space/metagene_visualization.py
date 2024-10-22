import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Example data
np.random.seed(42)
num_cells = 200
genes = ['Gene' + str(i) for i in range(1, 11)]
cell_types = ['TypeA', 'TypeB', 'TypeC']

# Create a random dataset
data = {
    'Cell': [f'Cell{i}' for i in range(num_cells)],
    'X': np.random.rand(num_cells) * 100,
    'Y': np.random.rand(num_cells) * 100,
    'CellType': np.random.choice(cell_types, num_cells),
}
# Add random gene expression data
for gene in genes:
    data[gene] = np.random.rand(num_cells) * 10

df = pd.DataFrame(data)

# Get the min and max values for the spatial coordinates
x_min, x_max = df['X'].min(), df['X'].max()
y_min, y_max = df['Y'].min(), df['Y'].max()

# Calculate the aspect ratio
aspect_ratio = (x_max - x_min) / (y_max - y_min)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Spatial Transcriptomics Data Visualization"),
    html.Div([
        dcc.Dropdown(
            id='cell-type-dropdown',
            options=[{'label': ct, 'value': ct} for ct in df['CellType'].unique()],
            value=df['CellType'].unique()[0],
            clearable=False,
            style={'width': '50%'}
        ),
        dcc.Dropdown(
            id='gene-dropdown',
            options=[{'label': gene, 'value': gene} for gene in genes],
            value=genes[0],
            clearable=False,
            style={'width': '50%'}
        ),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'padding': '20px'}),
    dcc.Graph(id='scatter-plot')
])

# Callback to update the scatter plot based on dropdown selection
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('cell-type-dropdown', 'value'),
     Input('gene-dropdown', 'value')]
)
def update_scatter_plot(selected_cell_type, selected_gene):
    # Filter the dataframe based on the selected cell type
    filtered_df = df[df['CellType'] == selected_cell_type]
    
    # Create the scatter plot
    fig = px.scatter(
        filtered_df,
        x='X', y='Y',
        color=selected_gene,
        hover_data=['Cell'],
        title=f'Spatial Plot of {selected_cell_type} - {selected_gene} Expression',
        labels={'X': 'Spatial X', 'Y': 'Spatial Y', selected_gene: 'Expression'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(range=[x_min, x_max], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[y_min, y_max], scaleanchor="x", scaleratio=1),
        width=800,  # Set the width of the figure
        height=800 / aspect_ratio  # Set the height based on the aspect ratio
    )

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
