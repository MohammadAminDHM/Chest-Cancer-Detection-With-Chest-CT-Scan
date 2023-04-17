import matplotlib.pyplot as plt
import json

metrics = ["loss", "accuracy", "recall", "precision", "f1"]
labels = ["train", "valid"]

plt.style.use("ggplot")
for metric in metrics:
    plt.figure()
    for label in labels:
        data = open(f"./model_ResNet50_{label}_{metric}_list.json")
        data = json.load(data)
        plt.plot(data, label=f"{label}_{metric}")
    plt.title(metric.capitalize())
    plt.xlabel("Epoch #")
    plt.ylabel(metric.capitalize())
    plt.legend(loc="lower right")
    plt.savefig(f"{metric.capitalize()}.png")
