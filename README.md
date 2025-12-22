This is just a basic tool that I will use to plot some of my experiements with. The main interface is in plot and is called `plot_metrics_and_runs`. This function takes in a list of metrics to graph and a list of runs to put on each graph. and does so via a list of `Metric` and `Run` objects. An example is shown in main. Feel free to edit this however and make it better. Below is an example image. I was trying to product something similar to wandb:

...

A couple of notes:
1. A weird parameter is `last_step_num`. Basically, you want this value to be as high as the number of step you have. For example, in my plots I take 100K steps so I put this value at 100100, slightly above 100K. If you put this value lower than the number of steps, the wandb api returns a strange subset of values. If your plots look strange and don't have all data, this value is likely the issue.
2. One weird trick I had to do was add space padding to the legend labels. For some reason no matter what I tried to do, the legend would always cutoff the test. I am guessing the method for calculating the output legend size for svgs is off by some factor of 2, but I don't have time to figure that out in the plotly codebase. Spaces work and nobody knows when they are looking at the final plot, so whatever >w<
