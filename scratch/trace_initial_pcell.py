import scipy.io
import numpy as np

sinr = scipy.io.loadmat('data/processed/sinr_30kmh_0.mat')['sinr']
if sinr.shape[0] == 5:
    sinr = sinr.T

# Find the best BS at step 0 (initial PCell)
best_initial_bs = np.argmax(sinr[0, :])
print(f"Step 0: Best initial BS is {best_initial_bs} with SINR {sinr[0, best_initial_bs]:.2f} dB")

# Trace the best initial BS's SINR values over time
for t in range(0, 1000, 50):
    print(f"Step {t}: BS {best_initial_bs} SINR = {sinr[t, best_initial_bs]:.2f} dB, Usable BSs: {np.where(sinr[t, :] > -8.0)[0]}")
