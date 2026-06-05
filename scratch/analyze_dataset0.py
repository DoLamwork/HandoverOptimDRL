import scipy.io
import numpy as np

# Load Dataset 0
sinr = scipy.io.loadmat('data/processed/sinr_30kmh_0.mat')['sinr']
# If sinr is (5, 5400)
if sinr.shape[0] == 5:
    n_bs, time_steps = sinr.shape
else:
    time_steps, n_bs = sinr.shape
    sinr = sinr.T

print(f"Dataset 0 has {time_steps} steps.")

# Let's find for each step, which BSs have SINR > -8.0 dB (usable)
usable_counts = np.sum(sinr > -8.0, axis=0)

# Let's print out if there are steps with very few usable BSs
for steps in [10, 50, 100, 200, 300, 400, 500, 600, 700]:
    outages = np.where(usable_counts == 0)[0]
    print(f"Total steps with 0 usable BS: {len(outages)}")
    
# Let's print the max SINR and usable BSs around step 600
print("\nAround step 600:")
for t in range(580, 620):
    best_bs = np.argmax(sinr[:, t])
    best_sinr = sinr[best_bs, t]
    print(f"Step {t}: Best BS {best_bs} with SINR {best_sinr:.2f} dB. Usable BSs: {np.where(sinr[:, t] > -8.0)[0]}")
