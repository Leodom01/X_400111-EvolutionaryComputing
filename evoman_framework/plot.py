import pandas as pd
import matplotlib.pyplot as plt

import os

data_dir = "./data.enemy3"
# data_dir = "./old.data/data"
# data_dir = "./data.enemy7"
# data_dir = "./data.enemy6"

experimens_directories = os.listdir(data_dir)

print(f"Found experiments: {experimens_directories}")

fig, axs = plt.subplots(3)

for experiment_name in experimens_directories:
	# if experiment_name not in ["2pt_crossover", 
	# 						   # "whole_arithmetic_recombination_07",
	# 						   # "random_arithmetic_recombination",
	# 						   "whole_arithmetic_recombination"]: continue

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

	def plot_column(axis, dfs, column):
		if dfs == []: return

		def compute_mean_std(dfs, column):
			df = pd.concat(dfs)    \
					.groupby(level=0) \
					.agg(avg=(column, "mean"), std=(column, "std"))
				
			return df

		def plot_mean_std(axis, df, name):
			df["avg"].plot(
				label=f"({experiment_name}) {name}",
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


	plot_column(axs[0], experiments, "mean_fitness")
	plot_column(axs[1], experiments, "max_fitness")
	# plot_column(axs[0], experiments, "min_fitness")
	# plot_column(axs[2], experiments, "std_fitness")
	plot_column(axs[2], experiments, "sq_disance_diversity")

fig.show()
plt.show()
