import csv
import matplotlib.pyplot as plt

steps, train, val = [],[],[]
with open("loss_log.csv") as f:
    for row in csv.DictReader(f):
        steps.append(int(row["steps"]))
        train.append(float(row["train"]))
        val.append(float(row["val"]))
plt.subplots(figsize=(7, 4))
plt.plot(steps,train, label="train")
plt.plot(steps, val, label="validation")
plt.xlabel("Step")
plt.ylabel("Cross-entropy loss")
plt.title("MiniGPT training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve_.png", dpi=150)