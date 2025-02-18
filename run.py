import subprocess

datasets = ["poetryfoundation"]
lrs = ["1e-3", "1e-4", "5e-4"]
models = ["lstm", "transformer"]

for dataset in datasets:
    for lr in lrs:
        for model in models:
            print(f"Running {model} on {dataset} with lr={lr} ...")
            command = ["python", "main.py", "--dataset", dataset, "--model", model, "--lr", lr]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
