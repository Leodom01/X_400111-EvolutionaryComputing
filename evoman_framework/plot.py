import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# data_dir = "./data.enemy3"
# data_dir = "./data.enemy7"
data_dir = "./data.enemy6"

font = {'size': 7}

plt.rc('font', **font)

experimens_directories = os.listdir(data_dir)

print(f"Found experiments: {experimens_directories}")

def prepare_fitness_plot(ax, title):
	plt.sca(ax)
	plt.grid(linestyle="-")
	plt.xlabel("Generation")
	plt.ylabel("Fitness")
	plt.yticks([ x * 10 for x in range(1, 11) ])
	plt.title(title)
	return ax

def prepare_diversity_plot(ax, title):
	plt.sca(ax)
	plt.xlabel("Generation")
	plt.ylabel("Diversity")
	plt.title(title)
	return ax

def prepare_box_plot(ax, title):
	plt.sca(ax)
	plt.title(title)
	return ax

fig, ((mean_plot, max_plot), (diversity_plot, box_plot)) = plt.subplots(ncols=2, nrows=2)
prepare_fitness_plot(mean_plot, "Mean fitness")
prepare_fitness_plot(max_plot, "Maximum fitness")
prepare_diversity_plot(diversity_plot, "Average diversity")
prepare_box_plot(box_plot, "Best individual fitness")

to_box_plot = []
box_plot_ticks = []

for experiment_name in experimens_directories:
	if experiment_name.endswith(".png"): continue
	if experiment_name.endswith(".pdf"): continue

	dir = f"{data_dir}/{experiment_name}"
	experiments_files = os.listdir(dir)
	experiments = []

	for experiment_file in experiments_files:
		if not experiment_file.endswith(".csv"): continue
		
		try:
			df = pd.read_csv(
				f"{dir}/{experiment_file}", 
				sep=", ", # type: ignore
				engine="python"
			)
			experiments.append(df)
		except:
			pass

	print(f"Loaded {len(experiments)} experiments in {experiment_name}")
	
	summary = np.loadtxt(f"{dir}/summary.txt")
	print(f"Loaded summary")

	def plot_column(axis, dfs, column):
		if dfs == []: return

		def compute_mean_std(dfs, column):
			df = pd.concat(dfs)    \
					.groupby(level=0) \
					.agg(avg=(column, "mean"), std=(column, "std"))
				
			return df

		def plot_mean_std(axis, df, name):
			df["avg"].plot(
				label=f"{experiment_name}",
				ax=axis,
				legend = True
			)

			axis.fill_between(
				df.index,
				df["avg"] - df["std"],
				df["avg"] + df["std"],
				alpha=0.2
			)

		data = compute_mean_std(dfs, column)
		plot_mean_std(axis, data, column)

	plot_column(mean_plot, experiments, "mean_fitness")
	plot_column(max_plot, experiments, "max_fitness")
	plot_column(diversity_plot, experiments, "sq_disance_diversity")
	to_box_plot.append(summary)
	box_plot_ticks.append(experiment_name)


plt.sca(box_plot)
plt.boxplot(to_box_plot)
box_plot.set_xticklabels(box_plot_ticks)

plt.tight_layout()
plt.savefig(f"{data_dir}/plot.pdf")
