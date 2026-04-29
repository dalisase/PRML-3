import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import os

# CUDA设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 参数设置
SEQ_LEN = 48
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.0005
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# 数据集处理
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "LSTM-Multivariate_pollution.csv")
test_path = os.path.join(BASE_DIR, "pollution_test_data1.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 风向编码
le = LabelEncoder()
train_df["wnd_dir"] = le.fit_transform(train_df["wnd_dir"])
test_df["wnd_dir"] = le.transform(test_df["wnd_dir"])

# 特征和标签准备
feature_cols = ["dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values
y_train = train_df["pollution"].values.reshape(-1, 1)
y_test  = test_df["pollution"].values.reshape(-1, 1)

# 标准化处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test  = scaler_y.transform(y_test)

# 滑动窗口处理
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)

X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE, shuffle=False)

# LSTM模型定义
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMPredictor(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练和测试循环
train_losses, test_losses = [], []
model.train()
for epoch in range(EPOCHS):
    # 训练
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 测试
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            pred = model(batch_x)
            total_test_loss += criterion(pred, batch_y).item()
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    model.train()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

# 评估最终模型
model.eval()
y_pred_scaled = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        pred = model(batch_x.to(DEVICE)).cpu()
        y_pred_scaled.append(pred)
y_pred = scaler_y.inverse_transform(torch.cat(y_pred_scaled).numpy())
y_true = scaler_y.inverse_transform(y_test_seq)

rmse = np.sqrt(np.mean((y_pred - y_true)**2))
print(f"\nFinal Test RMSE: {rmse:.4f}")

# 可视化结果
plt.figure(figsize=(14,5))

# 损失曲线
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss', color='b')
plt.plot(test_losses, label='Test Loss', color='r')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

# 预测绘图
plt.figure(figsize=(14,5))
sample_len = min(200, len(y_true))
plt.plot(y_true[:sample_len], label='True', color='g')
plt.plot(y_pred[:sample_len], 'r--', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('PM2.5')
plt.title(f'First {sample_len} Samples: True vs Predicted')
plt.legend()
plt.grid(True)
plt.show() 