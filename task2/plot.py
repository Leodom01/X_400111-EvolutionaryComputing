import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import os
font = {'size': 7}

plt.rc('font', **font)

def prepare_fitness_plot(ax, title):
  plt.sca(ax)
  plt.grid(linestyle="-")
  plt.xlabel("Generation")
  plt.ylabel("Fitness")
  # plt.yticks([ x * 10 for x in range(1, 11) ])
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

for enemy in ["1-2-3-4-5-6-7-8", "1-2-6"]:
  fig, ((mean_plot, max_plot), (diversity_plot, box_plot)) = plt.subplots(
    ncols=2, 
    nrows=2
  )

  prepare_fitness_plot(mean_plot, "Mean fitness ")
  prepare_fitness_plot(max_plot, "Maximum fitness")
  prepare_diversity_plot(diversity_plot, "Average diversity")
  prepare_box_plot(box_plot, "Best individual gain")

  gains_tot = []
  labels_tot = []

  for fitness in ["classic", "custom"]:
    dir = f"data/{enemy}/{fitness}"
    experiments_files = os.listdir(dir)
    experiments = []
    gains = []

    for experiment_file in experiments_files:
      if not experiment_file.endswith(".csv"): continue
      
      try:
        df = pd.read_csv(
          f"{dir}/{experiment_file}", 
          sep=",", # type: ignore
          engine="python"
        )
        experiments.append(df)
      except:
        pass
    
    for experiment_file in experiments_files:
      if not experiment_file.endswith(".gain"): continue
      gains.append(np.loadtxt(f"{dir}/{experiment_file}"))
    gains = list(map(list, zip(*gains)))

    def plot_column(axis, dfs, column):
      if dfs == []: return

      def compute_mean_std(dfs, column):
        df = pd.concat(dfs)    \
            .groupby(level=0) \
            .agg(avg=(column, "mean"), std=(column, "std"))
          
        return df

      def plot_mean_std(axis, df, name):
        df["avg"].plot(
          label=f"{fitness} {name}",
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

    plot_column(max_plot, experiments, "max_fitness_classic")
    plot_column(mean_plot, experiments, "mean_fitness_classic")
    plot_column(max_plot, experiments, "max_fitness_custom")
    plot_column(mean_plot, experiments, "mean_fitness_custom")
    plot_column(diversity_plot, experiments, "sq_distance_diversity")

    labels = enemy.split("-")
    labels = list(map(lambda en: f"{fitness} {en}", labels))
    
    gains_tot.append(list(map(lambda x: np.average(x), gains)))
    labels_tot.append(fitness)
    
  print(gains_tot)
  print(labels_tot)
  box_plot.boxplot(gains_tot, labels=labels_tot)

  
  plt.sca(box_plot)
  plt.tight_layout()
  plt.savefig(f"{enemy}.pdf")
