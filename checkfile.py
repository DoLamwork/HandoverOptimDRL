import scipy.io
import matplotlib.pyplot as plt

# 1. Đọc file .mat
mat_data = scipy.io.loadmat('data/processed/rsrp_30kmh_0.mat')
rsrp_matrix = mat_data['rsrp']  # Kích thước (5, 5400)

# 2. Thiết lập vẽ đồ thị
plt.figure(figsize=(15, 6))

# Vẽ đường tín hiệu của từng trạm phát
for i in range(5):
    plt.plot(rsrp_matrix[i, :], label=f'Trạm phát (BS) {i}')

# 3. Trang trí đồ thị
plt.title('Đồ thị cường độ sóng RSRP của 5 trạm phát (Toàn bộ 5400 bước)', fontsize=14)
plt.xlabel('Timestep (Bước thời gian)', fontsize=12)
plt.ylabel('Cường độ sóng RSRP (dBm)', fontsize=12)
plt.axvline(x=1950, color='red', linestyle='--', alpha=0.7, label='Vùng lõm sóng (Step 1950)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left')

# 4. Lưu đồ thị thành file ảnh để xem
output_image = 'rsrp_full_plot.png'
plt.savefig(output_image, dpi=300)
plt.close()

print(f" Đã vẽ xong đồ thị toàn bộ sóng và lưu thành ảnh: {output_image}")
print("Bạn hãy mở file ảnh này lên để xem bức tranh toàn cảnh của file dữ liệu!")