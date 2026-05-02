import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD METRICS
# =========================
df = pd.read_csv("../../data/results/metrics.csv")
# df = pd.read_csv("../multitask_metrics.csv")

models = df["Model"]


# =========================
# F1 PLOT
# =========================
def plot_F1(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))

    plt.bar(x, df["F1"], width=0.4)

    plt.xticks(x, df["Model"])
    plt.ylabel("F1 Score")
    plt.title("F1 Score Across Models")

    for i in range(len(df)):
        plt.text(i, df["F1"][i], f"{df['F1'][i]:.3f}", ha="center", va="bottom")

    plt.show()


# =========================
# FPR PLOT
# =========================
def plot_F1_class(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))

    plt.bar(x - 0.2, df["F1_Toxic"], width=0.4, label="Toxic")
    plt.bar(x + 0.2, df["F1_NonToxic"], width=0.4, label="Non-Toxic")

    plt.xticks(x, df["Model"])
    plt.ylabel("F1 Score")
    plt.title("F1 Score Across Models")

    plt.legend()

    for i in range(len(df)):
        plt.text(i - 0.2, df["F1_Toxic"][i], f"{df['F1_Toxic'][i]:.3f}", ha="center")
        plt.text(i + 0.2, df["F1_NonToxic"][i], f"{df['F1_NonToxic'][i]:.3f}", ha="center")

    plt.show()


# =========================
# FPR PLOT
# =========================
def plot_FPR(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))

    plt.bar(x - 0.2, df["FPR_AAE"], width=0.4, label="AAE")
    plt.bar(x + 0.2, df["FPR_SAE"], width=0.4, label="SAE")

    plt.xticks(x, df["Model"])
    plt.ylabel("False Positive Rate")
    plt.title("FPR Across Models")

    plt.legend()

    for i in range(len(df)):
        plt.text(i - 0.2, df["FPR_AAE"][i], f"{df['FPR_AAE'][i]:.3f}", ha="center")
        plt.text(i + 0.2, df["FPR_SAE"][i], f"{df['FPR_SAE'][i]:.3f}", ha="center")

    plt.show()


# =========================
# FNR PLOT
# =========================
def plot_FNR(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))

    plt.bar(x - 0.2, df["FNR_AAE"], width=0.4, label="AAE")
    plt.bar(x + 0.2, df["FNR_SAE"], width=0.4, label="SAE")

    plt.xticks(x, df["Model"])
    plt.ylabel("False Negative Rate")
    plt.title("FNR Across Models")

    plt.legend()

    for i in range(len(df)):
        plt.text(i - 0.2, df["FNR_AAE"][i], f"{df['FNR_AAE'][i]:.3f}", ha="center")
        plt.text(i + 0.2, df["FNR_SAE"][i], f"{df['FNR_SAE'][i]:.3f}", ha="center")

    plt.show()


# =========================
# DI PLOT
# =========================
def plot_DI(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))

    plt.bar(x - 0.2, df["DIfav"], width=0.4, label="DIfav (Non-Toxic)")
    plt.bar(x + 0.2, df["DIunfav"], width=0.4, label="DIunfav (Toxic)")

    plt.axhline(1.0, linestyle="--")

    plt.xticks(x, df["Model"])
    plt.ylabel("Disparate Impact")
    plt.title("Disparate Impact Across Models")

    plt.legend()

    for i in range(len(df)):
        plt.text(i - 0.2, df["DIfav"][i], f"{df['DIfav'][i]:.2f}", ha="center")
        plt.text(i + 0.2, df["DIunfav"][i], f"{df['DIunfav'][i]:.2f}", ha="center")

    plt.show()


# =========================
# RUN ALL PLOTS
# =========================
plot_F1(df)
plot_F1_class(df)
plot_FPR(df)
plot_FNR(df)
plot_DI(df)
