import wandb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots
import os

# Initialize the API
api = wandb.Api()




# Metrics to get
desired_metrics = [
    "test_loss",
    "loss",
]
# Names for the graphs of each metric (respectively)
desired_metrics_rename = [
    "test loss",
    "train loss",
]
# Transforms to apply to each metric
transforms = [
    None,
    None,
]
# Title for each graph
graph_titles = [
    "Test Loss",
    "Train Loss"
]
# Output directories for each graph
graph_dirs = [
    "output",
    "output"
]
# Output filenames for each graph (probably want a svg)
graph_filename = [
    "test_loss.svg",
    "train_loss.svg"
]

# Runs we want in this graph by name or id
runs_we_want = [
    "small_8192sl_gpu_32bs__softmax",
    "small_8192sl_gpu_32bs__mamba"
]
# Names of the runs on the graph
run_names = [
    "Softmax",
    "Mamba"
]
# Colors for each run on the graph
run_colors = [
    "blue",
    "red"
]

# The number of the last step. Without this value, scan_history
# returns a weird subset of data. As long as exp_num_vals > the last step
# you want to get, I think it should be fine.
last_step_num = 100_100
# Name to rename the x axis (_step)
x_axis_rename = "step"
# Put axis names?
axis_names = True


# Get all train runs
project_path = "gmongaras1/Mamba_Squared_Experiemnts" # entity/project_name
runs = api.runs(project_path)

# Graphs for each variable we want to plot
figs = {
    k: []
    for k in desired_metrics
}

# The run names are really annoying with padding. To artificially pad them, I'm gonna add spaces
run_names = [i + " "*(len(i)//2) for i in run_names]

# Iterate over all runs to get the data from them
for run in runs:
    # Skip runs we don't want
    if run.name in runs_we_want:
        run_name = run.name
    elif run.id in runs_we_want:
        run_name = runs.id
    else:
        continue
    
    # Color for this run
    run_idx = runs_we_want.index(run_name)
    run_color = run_colors[run_idx]
    run_rename = run_names[run_idx]
    
    # Extract all the data for a specific set of metrics
    history = pd.DataFrame([i for i in run.scan_history(page_size=last_step_num)])
    
    # Get the step metric for the x axis of all plots
    steps = np.array(history["_step"])
    
    # Get all keys we want to plot plus the step
    history = history[desired_metrics + ["_step"]]
    
    # Add graphs for each variable
    for var, var_name in zip(desired_metrics, desired_metrics_rename):
        data_ = history[["_step", var]]
        data_ = data_[~data_[var].isna()]
        figs[var].append(
            go.Scatter(
                x=data_["_step"],
                y=data_[var],
                name=run_rename,
                marker=dict(
                    color=run_color,
                    line=dict(
                        color=run_color,
                        width=1
                    )
                ),
                mode="lines"
            )
        )

# Save graphs
for var, graph_name, var_name, dir_, filename in zip(desired_metrics, graph_titles, desired_metrics_rename, graph_dirs, graph_filename):
    # Combine plots
    fig = subplots.make_subplots(
        rows=1,
        cols=1, 
        vertical_spacing=0.5,
        x_title=x_axis_rename if axis_names else None,
        y_title=var_name if axis_names else None,
    )
    for trace in figs[var]:
        fig.add_trace(trace)
    
    # Reformat
    fig.update_layout(
        height=600,
        width=1200,
        title=dict(
            text=graph_name,
            x=0.5,
            y=0.96, # Position the title near the top
            xanchor="center",
            yanchor="top",
            font_color="rgb(0, 0, 0)",
            font_size=25,
        ),
        # legend={
        #     'x': 0.5, # Center the legend horizontally
        #     'y': 1.0, # Place the legend just below the title/top margin
        #     'xanchor': 'center',
        #     'yanchor': 'bottom', # Anchor the bottom of the legend box to the y coordinate
        #     'orientation': 'h' # Display legend items horizontally
        # },
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font_size=15,
            itemwidth=60,
        ),
        # Remove verticle lines
        xaxis=dict(
            showgrid=False,
            showline=True,
            zeroline=False,
            linecolor="LightGrey",
            color="DarkGrey",
            linewidth=2,
            ticks="outside",
            tickfont_color="#2a3f5f",
        ),
        # Show horizontal lines
        yaxis=dict(
            showgrid=True,
            showline=True,
            zeroline=False,
            linecolor="LightGrey",
            gridcolor="#e5e8ef",
            color="DarkGrey",
            linewidth=2,
            ticks="outside",
            tickfont_color="#2a3f5f",
        ),
        # White background
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        # Margin
        margin=dict(l=70, r=30, t=50, b=60, pad=4),
        overwrite=True,
    )
    
    # Size of the axes
    fig.update_annotations(font_size=20)
    
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig.write_image(os.path.join(dir_, filename))
