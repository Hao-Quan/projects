import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

model = torch.load("results/NTU/SGN/1_best.pth")
model.eval()



#confusion_matrix = confusion_matrix(y_test, predictions).astype(np.float)

print("")