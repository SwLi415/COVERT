import matplotlib.pyplot as plt
import scienceplots
import os

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({
    "font.family": "serif",  # specify font family here
    "font.serif": ["Times New Roman"],  # specify font here
    "font.size": 8})  # specify font size here

fig = plt.figure(figsize=(9, 9), dpi=100)
epoch = ['1', '5', '10', '15', '20', '25', '30', '35', '40']
accuracy_noniid = [8.00, 34.92, 53.45, 68.48, 63.35, 63.53, 62.08, 71.28, 77.97]
accuracy_iid = [10.27, 11.55, 93.50, 94, 94.20, 94.05, 94.57, 94.55, 94.63]

plt.plot(epoch, accuracy_noniid, c='green', label='non-iid')
plt.plot(epoch, accuracy_iid, c='blue', label='iid')

print(os.getcwd())
plt.savefig('test.pdf')
# plt.show()
