

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

data = {
    'GE': [0.3987, 0.4223, 0.4659, 0.4138, 0.3403],
    'CP': [0.4657, 0.4602, 0.4802, 0.5152, 0.5173],
    'CS': [0.5244, 0.5863, 0.8357, 0.5909, 0.5964],
}

# Convert the data into a pandas DataFrame with long format
long_data = pd.DataFrame(data).melt(var_name='Algorithm', value_name='Macro F1 Score')

# Define colors
colors = ['orange', 'green', 'red'] # You can adjust these to whatever colors you prefer

# Create figure
fig = go.Figure()

for algorithm, color in zip(long_data['Algorithm'].unique(), colors):
    fig.add_trace(go.Box(
        y=long_data['Macro F1 Score'][long_data['Algorithm'] == algorithm],
        name=algorithm,
        marker_color=color,
    ))

# Update layout and axes
fig.update_layout(
    title='Performance of Algorithms',
    yaxis_title='Macro F1 Score',
    xaxis_title='Algorithm',
    plot_bgcolor='white',  # set the background color to white
    showlegend=False,  # hide the legend
    autosize=False,  # allow manual sizing of the figure
    width=800,  # width of the figure in pixels
    height=600,  # height of the figure in pixels
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor='black',  # Set x-axis line color
        tickmode='array',  # Use tick array
        tickvals=list(range(len(data))),  # Set tick positions
        ticktext=list(data.keys()),  # Set tick labels
        tickangle=30,  # Rotate x-axis labels by 45 degrees
        ticks='outside',  # Place ticks outside the plot
        automargin=True  # Shrink margins to fit
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor='black',  # Set y-axis line color
        tickmode='auto',  # Use automatic tick placement
        ticks='outside',  # Place ticks outside the plot
        range=[0, 1]  # set y range from 0 to 1
    )
)
 # Customize the title's font
fig.update_layout(title_font=dict(size=24, family='Times New Roman', color='black'))

# Customize the axis labels' font
fig.update_xaxes(title_font=dict(size=18, family='Times New Roman', color='black'))
fig.update_yaxes(title_font=dict(size=18, family='Times New Roman', color='black'))

# Customize the tick labels' font and color
fig.update_xaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))
fig.update_yaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))

# Set axis line color
fig.update_xaxes(showline=True, linecolor='black')
fig.update_yaxes(showline=True, linecolor='black')

    # Add ticks that go through the axis
fig.update_xaxes(ticks="inside")
fig.update_yaxes(ticks="inside")


fig.show()

# Data
data = {
    'CP + GE: Soft-Voting': [0.0698, 0.1802, 0.2729, 0.1968, 0.1265],
    'CP + GE: End-to-End': [0.284413, 0.429403, 0.408006, 0.342319, 0.309288],
    'CP + CS: Soft-Voting': [0.5113, 0.6017, 0.8360, 0.5879, 0.5680],
    'CP + CS: End-to-End': [0.48112, 0.62587, 0.711786, 0.470741, 0.492986],
    'CS + GE: Soft-Voting': [0.4128, 0.6291, 0.7890, 0.5460, 0.4970],
    'CS + GE: End-to-End': [0.470741, 0.492986, 0.631022, 0.48771, 0.446057],
    'CP + GE + CS: Soft-Voting': [0.5525, 0.6226, 0.6966, 0.5891, 0.5514],
    'CP + GE + CS: End-to-End': [0.457455, 0.518991, 0.5675, 0.4962, 0.670056],
}

# Convert the data into a pandas DataFrame with long format
long_data = pd.DataFrame([
    {
        'Algorithm': k.split(':')[0],
        'Method': k.split(':')[1].strip(),
        'Performance': val
    }
    for k, values in data.items() for val in values
])
fig = px.box(long_data, x="Algorithm", y="Performance", color="Method", title="Performance of Algorithms", range_y=[0,1])
# Update layout and axes
fig.update_layout(
    title='Performance of Algorithms',
    yaxis_title='Performance',
    xaxis_title='Algorithm',
    plot_bgcolor='white',  # set the background color to white
    showlegend=True,  # hide the legend
    autosize=False,  # allow manual sizing of the figure
    width=800,  # width of the figure in pixels
    height=600,  # height of the figure in pixels
    xaxis=dict(
        showline=True,  # Show x-axis line
        linecolor='black',  # Set x-axis line color
        tickmode='array',  # Use tick array
        tickvals=list(range(len(data))),  # Set tick positions
        tickangle=30,  # Rotate x-axis labels by 45 degrees
        ticks='outside',  # Place ticks outside the plot
        automargin=True  # Shrink margins to fit
    ),
    yaxis=dict(
        showline=True,  # Show y-axis line
        linecolor='black',  # Set y-axis line color
        tickmode='auto',  # Use automatic tick placement
        ticks='outside',  # Place ticks outside the plot
        range=[0, 1]  # set y range from 0 to 1
    )
)
 # Customize the title's font
fig.update_layout(title_font=dict(size=24, family='Times New Roman', color='black'))

# Customize the axis labels' font
fig.update_xaxes(title_font=dict(size=18, family='Times New Roman', color='black'))
fig.update_yaxes(title_font=dict(size=18, family='Times New Roman', color='black'))

# Customize the tick labels' font and color
fig.update_xaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))
fig.update_yaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))

# Set axis line color
fig.update_xaxes(showline=True, linecolor='black')
fig.update_yaxes(showline=True, linecolor='black')

    # Add ticks that go through the axis
fig.update_xaxes(ticks="inside")
fig.update_yaxes(ticks="inside")

fig.show()
