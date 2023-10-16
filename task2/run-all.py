import os

for enemy in ["1-2-3-4-5-6-7-8", "1-2-6"]:
    for fitness in ["classic", "custom"]:
        for run in range(1):
            print(f"Run {fitness} {enemy} {run}")
            os.environ["FITNESS_FUNCTION"] = fitness
            os.environ["N_RUN"] = str(run)
            os.environ["ENEMIES"] = enemy
            
            os.system("python train.py")
            print(f"Run {fitness} {enemy} {run} done")
