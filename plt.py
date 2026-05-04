import numpy as np
import matplotlib.pyplot as plt

np.random.seed(62)

epochs = np.arange(1, 201)

# --- Type-I: clear decreasing trend ---
trend_I = np.linspace(0.65, 0.45, len(epochs))   # strong downward trend
noise_I = np.random.normal(0, 0.05, len(epochs)) # small randomness
type_I = trend_I + noise_I
type_I = np.clip(type_I, 0.1, 0.7)

# --- Type-II: small, random ---
type_II = np.random.uniform(0.05, 0.17, size=len(epochs))

# --- Plot ---
plt.figure(figsize=(8, 4.5))
plt.plot(epochs, type_I, label="Type-I Error (False Trigger)", linewidth=2)
plt.plot(epochs, type_II, label="Type-II Error (Mis Trigger)", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title("Errors over Epochs")
plt.ylim(0, 0.75)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("error_rates.png", dpi=300)