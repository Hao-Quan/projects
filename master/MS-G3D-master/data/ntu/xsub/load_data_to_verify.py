import numpy as np
import pickle as pkl

train_data = np.load("train_data_joint.npy")
test_data = np.load("val_data_joint.npy")

with open('train_label_2.pkl', 'rb') as f:
    train_label = pkl.load(f)

with open('val_label_2.pkl', 'rb') as f:
    test_label = pkl.load(f)

# # x = [train_data[i] for i in range(len(train_data)) if i % 2 == 0]
# # y = [train_data[i] for i in range(len(train_data)) if i % 2 != 0]
# x = []
# y = []
# z = []
#
# for i in range(len(train_data)):
#     x.append([train_data[i][k] for k in range(36) if k % 2 == 0])
#     y.append([train_data[i][k] for k in range(36) if k % 2 != 0])
# z.append(x)
# z.append(y)
# z = np.array(z)
#
#
#
# for i in range(len(test_data)):
#     x.append([test_data[i][k] for k in range(36) if k % 2 == 0])
#     y.append([test_data[i][k] for k in range(36) if k % 2 != 0])
# z.append(x)
# z.append(y)
# z = np.array(z)
#
#
print("")