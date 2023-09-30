# GRP 22

Evoman assignment 1

## How is structured

In each of the following folders: 

```bash
data.enemy3
data.enemy6
data.enemy7
```
we can find the:

```python
# -pdf containing min_fitness, max_fitness, average_diversity and best individusal gain boxplot
plot.pdf

# -two point crossover folder containing:
# "individual-N".txt showing the fitnesses of the variuos weiights of each individual
2pt_crossover/"individual-N".txt
# "run-N".csv showing csv of each run with relative data
2pt_crossover/"run-N".csv

# -whole arithmetic recombination folder containing:
# "individual-N".txt showing the fitnesses of the variuos weiights of each individual
whole_arithmetic_recombination/"individual-N".txt
# "run-N".csv showing csv of each run with relative data
whole_arithmetic_recombination//"run-N".csv

```
to run experiment:

```python
# to run experiment
python3  train.py

# run only plot
python3 plot.py

# to test
python3 test.py
```
## License

[MIT](https://choosealicense.com/licenses/mit/)
