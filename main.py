from plot import Metric, Run, TableInfo, plot_metrics_and_runs


# Metrics to plot
metrics = [
    Metric(
        metric_name="loss",
        metric_name_plot="train loss",
        metric_graph_title="Train Loss",
        metric_graph_out_dir="output",
        metric_graph_filename="train_loss.svg",
        metric_n_step_avg=1000
    ),
    Metric(
        metric_name="test_loss",
        metric_name_plot="test loss",
        metric_graph_title="Test Loss",
        metric_graph_out_dir="output",
        metric_graph_filename="test_loss.svg"
    ),
]

# Runs to plot
runs = [
    Run(
        run_name="small_8192sl_gpu_32bs__softmax",
        run_name_plot="Softmax",
        run_color="blue"
    ),
    Run(
        run_name="small_8192sl_gpu_32bs__mamba",
        run_name_plot="Mamba",
        run_color="red"
    )
]

plot_metrics_and_runs(
    metrics=metrics,
    runs=runs,
    project_entity="gmongaras1",
    project_name="Mamba_Squared_Experiemnts",
    last_step_num=100_100,
    x_axis_rename="step",
    plot_axis_names=True,
    table_info=TableInfo(
        table_filename="table.txt",
        bold_best=True,
        num_decimals=2
    ),
)