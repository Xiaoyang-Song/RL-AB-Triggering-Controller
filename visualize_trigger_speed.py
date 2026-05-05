import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the evaluation results
df = pd.read_csv('results/evaluation_results_b15.0_c16.0_b25.0_c25.0_c35.0_eta0.2.csv')

# Set random seed for reproducible jitter
np.random.seed(42)

# Jitter amount
jitter = 0.1

# Plot 1: Speed at Trigger
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Triggered + Collision
tc_trigger = df[(df['collision'] == True) & (df['triggered'] == True)]['speed_at_trigger'].dropna()
x_tc = 0 + np.random.uniform(-jitter, jitter, len(tc_trigger))
ax1.scatter(x_tc, tc_trigger, color='green', label='Triggered + Collision', alpha=0.7, s=50)

# False Trigger
ft_trigger = df[(df['collision'] == False) & (df['triggered'] == True)]['speed_at_trigger'].dropna()
x_ft = 1 + np.random.uniform(-jitter, jitter, len(ft_trigger))
ax1.scatter(x_ft, ft_trigger, color='red', label='False Trigger', alpha=0.7, s=50)

ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Triggered\n+ Collision', 'False\nTrigger'])
ax1.set_xlabel('Case')
ax1.set_ylabel('Speed at Trigger (km/h)')
ax1.set_title('Speed at Trigger by Case')
ax1.legend()
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig('speed_at_trigger.png', dpi=300)
# plt.show()

# Plot 2: Speed at Collision
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Triggered + Collision
tc_collision = df[(df['collision'] == True) & (df['triggered'] == True)]['speed_at_collision'].dropna()
x_tc2 = 0 + np.random.uniform(-jitter, jitter, len(tc_collision))
ax2.scatter(x_tc2, tc_collision, color='green', label='Triggered + Collision', alpha=0.7, s=50)

# Missed Collision
mc_collision = df[(df['collision'] == True) & (df['triggered'] == False)]['speed_at_collision'].dropna()
x_mc = 1 + np.random.uniform(-jitter, jitter, len(mc_collision))
ax2.scatter(x_mc, mc_collision, color='blue', label='No Trigger + Collision', alpha=0.7, s=50)

ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Triggered\n+ Collision', 'No-Trigger\n+Collision'])
ax2.set_xlabel('Case')
ax2.set_ylabel('Speed at Collision (km/h)')
ax2.set_title('Speed at Collision by Case')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('speed_at_collision.png', dpi=300)
# plt.show()

