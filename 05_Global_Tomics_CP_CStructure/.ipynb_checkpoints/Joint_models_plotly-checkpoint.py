import pandas as pd
import plotly.graph_objects as go

# Sample data
data = {
    'Gradient Boosting': [0.3643, 0.3558, 0.3720, 0.3395, 0.3639],
    'K Nearest Neighbour': [0.3578, 0.2669, 0.3624, 0.3156, 0.2798],
    'Bagging Classifier': [0.3098, 0.3024, 0.3089, 0.2834, 0.3088],
    'Simple NN': [0.4360, 0.3446, 0.4252, 0.3948, 0.3304],
    'Simple NN + Cell Line': [0.4137, 0.3682, 0.4435, 0.3824, 0.3749],
    '1D CNN': [0.4170, 0.3543, 0.4638, 0.4073, 0.3676],
    '1D CNN + Cell Line': [0.3987, 0.4223, 0.4659, 0.4138, 0.3403],
    'DeepInsight': [0.4384, 0.3641, 0.4648, 0.3572, 0.3754],
    'IGTD': [0.3609, 0.4317, 0.4500, 0.3686, 0.4013],
    'DWTM': [0.2798, 0.1930, 0.2631, 0.2358, 0.2544],
    'Chemical Structure': [0.5244, 0.5863, 0.8357, 0.5909, 0.5964],
    'Cell Painting': [0.4657, 0.4602, 0.4802, 0.5152, 0.5173]
}

# Convert the data into a pandas DataFrame with long format
long_data = pd.DataFrame(data).melt(var_name='Algorithm', value_name='Performance')

# Define colors
colors = ['orange', 'green', 'red', 'blue', 'purple', 'cyan', 'magenta', 'lime', 'yellow', 'pink', 'brown', 'gray']

# Create figure
fig = go.Figure()

# Add a box plot for each algorithm
for algorithm, color in zip(long_data['Algorithm'].unique(), colors):
    fig.add_trace(go.Box(
        y=long_data[long_data['Algorithm'] == algorithm]['Performance'],
        name=algorithm,
        marker_color=color
    ))

# Update layout and axes
fig.update_layout(
    title='Performance of 12 Algorithms on erik_hq_8_12',
    yaxis_title='Macro F1 Score',
    xaxis_title='Algorithm',
    plot_bgcolor='white',
    showlegend=False,
    autosize=False,
    width=800,
    height=600,
    xaxis=dict(
        showline=True,
        linecolor='black',
        tickmode='array',
        tickvals=list(range(len(data))),
        ticktext=list(data.keys()),
        tickangle=30,
        ticks='outside',
        automargin=True,
        tickfont=dict(size=14, family='Times New Roman', color='black')
    ),
    yaxis=dict(
        showline=True,
        linecolor='black',
        tickmode='auto',
        ticks='outside',
        range=[0, 1],
        tickfont=dict(size=14, family='Times New Roman', color='black')
    ),
    title_font=dict(size=24, family='Times New Roman', color='black')
)

# Show the plot
fig.show()
