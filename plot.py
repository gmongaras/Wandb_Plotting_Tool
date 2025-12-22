import wandb
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots
import os
from dataclasses import dataclass
from typing import Callable

# Initialize the API
api = wandb.Api()

# Identity function is the default
identity = lambda x: x

# Used to store all information about a metric we want to create a graph for
@dataclass
class Metric:
    metric_name: str # wandb name for the metric to plot
    metric_name_plot: str # Name of the metric on the plot
    metric_graph_title: str # Title for the graph of this metric
    metric_graph_out_dir: str # Output directory for the graph
    metric_graph_filename: str # Output filename for the graph
    metric_transform: Callable = identity # Function used to transform data before plotting

# Used to store all information about a run we want to graph
@dataclass
class Run:
    run_name: str # wandb name of the run to plot on each graph
    run_name_plot: str # Name of the run on the plot
    run_color: str # Color to plot for this run



def plot_metrics_and_runs(
        # List of metrics for each graph to plot
        metrics: list[Metric],
        # List of runs to plot on each graph
        runs: list[Run], 
        # Entity of your project
        project_entity: str,
        # Name of your project within the entity
        project_name: str,
        # The number of the last step. Without this value, scan_history
        # returns a weird subset of data. As long as exp_num_vals > the last step
        # you want to get, I think it should be fine.
        last_step_num: int = 100_100,
        # Name to rename the x axis (_step)
        x_axis_rename: str = "step",
        # Put axis names?
        plot_axis_names: bool = True,
    ):
    
    # All run names we want to plot
    runs_we_want = [run.run_name for run in runs]
    
    # All metric names and transforms we want to create plots of
    desired_metrics = [metric.metric_name for metric in metrics]
    transforms = [metric.metric_transform for metric in metrics]
    
    # Get all train runs
    project_path = f"{project_entity}/{project_name}"
    all_project_runs = api.runs(project_path)

    # Graphs for each variable we want to plot
    figs = {m:[] for m in desired_metrics}
    # Iterate over all runs to get the data from them
    for project_run in all_project_runs:        
        # Skip runs we don't want
        if project_run.name in runs_we_want:
            run_name = project_run.name
        elif project_run.id in runs_we_want:
            run_name = runs.id
        else:
            continue
        
        # If this a run we actually want to look at, get the variable for
        # plotting this run.
        run_idx = runs_we_want.index(run_name)
        run_plot_vars = runs[run_idx]
        run_name = run_plot_vars.run_name
        run_name_plot = run_plot_vars.run_name_plot
        run_color = run_plot_vars.run_color
        
        # The run names are really annoying with padding. To artificially pad them, I'm gonna add spaces
        run_name_plot = run_name_plot + " "*(len(run_name_plot)//2)
        
        # Extract all the data for a specific set of metrics
        history = pd.DataFrame([i for i in project_run.scan_history(page_size=last_step_num)])
        
        # Get all keys we want to plot plus the step
        history = history[desired_metrics + ["_step"]]
        
        # Add graphs for each variable
        for var, transform in zip(desired_metrics, transforms):
            data_ = history[["_step", var]]
            data_ = data_[~data_[var].isna()]
            figs[var].append(
                go.Scatter(
                    x=data_["_step"],
                    y=transform(data_[var]),
                    name=run_name_plot,
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
    for metric in metrics:
    # for var, graph_name, var_name, dir_, filename in zip(desired_metrics, graph_titles, desired_metrics_rename, graph_dirs, graph_filename):
        var = metric.metric_name
        var_name = metric.metric_name_plot
        title = metric.metric_graph_title
        dir_ = metric.metric_graph_out_dir
        filename = metric.metric_graph_filename
        
        # Combine plots
        fig = subplots.make_subplots(
            rows=1,
            cols=1, 
            vertical_spacing=0.5,
            x_title=x_axis_rename if plot_axis_names else None,
            y_title=var_name if plot_axis_names else None,
        )
        for trace in figs[var]:
            fig.add_trace(trace)
        
        # Reformat
        fig.update_layout(
            height=600,
            width=1200,
            title=dict(
                text=title,
                x=0.5,
                y=0.96, # Position the title near the top
                xanchor="center",
                yanchor="top",
                font_color="rgb(0, 0, 0)",
                font_size=25,
            ),
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
