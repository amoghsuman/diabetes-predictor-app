# visualizations 

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import numpy as np

def generate_pairplot(data, output_path="images/pairplot.png"):
    sns.set_context("paper", rc={"axes.labelsize":18})
    plot = sns.pairplot(data, hue='Outcome', palette='Set2', corner=True, height=1.5)
    for ax in plot.axes.flatten():
        if ax:
            ax.set_xlabel(ax.get_xlabel(), rotation=-55, horizontalalignment='left')
            ax.set_ylabel(ax.get_ylabel(), rotation=-55, horizontalalignment='right')
    plot.savefig(output_path)
    plt.close()


#Task 5A: Correlation Matrix (Heatmap)
#reusable visualization
def plot_correlation_matrix(data, output_path="images/corr_heatmap.png"):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr[(corr >= 0.2) | (corr <= -0.2)],
        cmap='viridis',
        vmax=1.0,
        vmin=-1.0,
        linewidths=0.1,
        annot=True,
        annot_kws={"size": 8},
        square=True
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#Task 6B: Distribution Plots
def plot_feature_distributions(data, output_path="images/distributions.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rcParams

    feature_names = list(data.columns)[:8]
    rcParams['figure.figsize'] = 20, 15
    sns.set(font_scale=1)
    sns.set_style("white")
    sns.set_palette("bright")

    plt.subplots_adjust(hspace=0.5)
    for i, name in enumerate(feature_names):
        plt.subplot(4, 2, i + 1)
        sns.histplot(data=data, x=name, hue="Outcome", kde=True, palette="BuGn")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#Task 10: Printing the Training Curves
def plot_training_curves(history, output_path="images/training_curves.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["figure.figsize"] = (12, 8)
    N = np.arange(0, len(history.history["loss"]))
    plt.style.use("ggplot")
    
    plt.figure()
    plt.plot(N, history.history["loss"], label="Train Loss")
    plt.plot(N, history.history["val_loss"], label="Val Loss")
    plt.plot(N, history.history["accuracy"], label="Train Accuracy")
    plt.plot(N, history.history["val_accuracy"], label="Val Accuracy")
    
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#Task 11: Function to plot Confusion Matrix
def plot_confusion_matrix(Y_test, Y_pred_probs, output_path="images/confusion_matrix.png"):
    actuals = np.argmax(Y_test.to_numpy().T, axis=0)
    predicted = np.argmax(Y_pred_probs, axis=1)

    cm = confusion_matrix(actuals, predicted)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
