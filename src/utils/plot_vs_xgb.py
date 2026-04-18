import os
import pandas as pd
import matplotlib.pyplot as plt

# Load candidate results
df = pd.read_csv("data/results/vs_xgb_train_time/vs_xgb_candidates.csv")


# PLOTTING ALPHA
# Fix alpha to the best one (or whichever one you want to show)
alpha_fixed = 1.0
plot_df = df[df["alpha_aae"] == alpha_fixed].sort_values("beta_aae")

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- FPR vs beta ---
axes[0, 0].plot(plot_df["beta_aae"], plot_df["FPR_AAE"], marker="o", label="FPR AAE")
axes[0, 0].plot(plot_df["beta_aae"], plot_df["FPR_SAE"], marker="o", label="FPR SAE")
axes[0, 0].plot(plot_df["beta_aae"], plot_df["FPR_gap"], marker="o", label="FPR gap")
axes[0, 0].set_title(f"FPR vs Beta (alpha={alpha_fixed})")
axes[0, 0].set_xlabel("beta_aae")
axes[0, 0].set_ylabel("FPR")
axes[0, 0].legend()

# --- Performance vs beta ---
axes[0, 1].plot(plot_df["beta_aae"], plot_df["f1"], marker="o", label="F1")
axes[0, 1].plot(plot_df["beta_aae"], plot_df["accuracy"], marker="o", label="Accuracy")
axes[0, 1].set_title(f"Performance vs Beta (alpha={alpha_fixed})")
axes[0, 1].set_xlabel("beta_aae")
axes[0, 1].set_ylabel("Score")
axes[0, 1].legend()

# --- FNR vs beta ---
axes[1, 0].plot(plot_df["beta_aae"], plot_df["FNR_AAE"], marker="o", label="FNR AAE")
axes[1, 0].plot(plot_df["beta_aae"], plot_df["FNR_SAE"], marker="o", label="FNR SAE")
axes[1, 0].plot(plot_df["beta_aae"], plot_df["FNR_gap"], marker="o", label="FNR gap")
axes[1, 0].set_title(f"FNR vs Beta (alpha={alpha_fixed})")
axes[1, 0].set_xlabel("beta_aae")
axes[1, 0].set_ylabel("FNR")
axes[1, 0].legend()

# Empty bottom-right subplot if you want to keep same layout as slide
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("src/plots/vs_xgb_beta_plots.png", dpi=300, bbox_inches="tight")
plt.show()




# PLOTTING BETA
# Fix beta at -1.0
beta_fixed = -1.0
plot_df = df[df["beta_aae"] == beta_fixed].sort_values("alpha_aae")

# Make figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- FPR vs alpha ---
axes[0, 0].plot(plot_df["alpha_aae"], plot_df["FPR_AAE"], marker="o", label="FPR AAE")
axes[0, 0].plot(plot_df["alpha_aae"], plot_df["FPR_SAE"], marker="o", label="FPR SAE")
axes[0, 0].plot(plot_df["alpha_aae"], plot_df["FPR_gap"], marker="o", label="FPR gap")
axes[0, 0].set_title(f"FPR vs Alpha (beta={beta_fixed})")
axes[0, 0].set_xlabel("alpha_aae")
axes[0, 0].set_ylabel("FPR")
axes[0, 0].legend()

# --- Performance vs alpha ---
axes[0, 1].plot(plot_df["alpha_aae"], plot_df["f1"], marker="o", label="F1")
axes[0, 1].plot(plot_df["alpha_aae"], plot_df["accuracy"], marker="o", label="Accuracy")
axes[0, 1].set_title(f"Performance vs Alpha (beta={beta_fixed})")
axes[0, 1].set_xlabel("alpha_aae")
axes[0, 1].set_ylabel("Score")
axes[0, 1].legend()

# --- FNR vs alpha ---
axes[1, 0].plot(plot_df["alpha_aae"], plot_df["FNR_AAE"], marker="o", label="FNR AAE")
axes[1, 0].plot(plot_df["alpha_aae"], plot_df["FNR_SAE"], marker="o", label="FNR SAE")
axes[1, 0].plot(plot_df["alpha_aae"], plot_df["FNR_gap"], marker="o", label="FNR gap")
axes[1, 0].set_title(f"FNR vs Alpha (beta={beta_fixed})")
axes[1, 0].set_xlabel("alpha_aae")
axes[1, 0].set_ylabel("FNR")
axes[1, 0].legend()

# Empty bottom-right panel to match the other layout
axes[1, 1].axis("off")

plt.tight_layout()

# Save
os.makedirs("src/plots", exist_ok=True)
plt.savefig("src/plots/vs_xgb_alpha_plots.png", dpi=300, bbox_inches="tight")
plt.show()