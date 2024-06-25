import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generowanie syntetycznych danych
def generate_synthetic_data(seq_len, num_samples):
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 10, seq_len)
        x = 10 * np.exp(-0.1 * t) * np.cos(t)  # Pozycja z tłumieniem
        v = -10 * np.exp(-0.1 * t) * np.sin(t)  # Prędkość z tłumieniem
        sample = np.stack((x, v), axis=1)
        data.append(sample)
    return np.array(data)

seq_len = 100
num_samples = 1000
data = generate_synthetic_data(seq_len, num_samples)

# Przygotowanie zbioru danych
class BallTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx + 1][-1]

dataset = BallTrajectoryDataset(data, seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Definicja modelu RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 2
hidden_size = 50
output_size = 2
num_layers = 2

model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modelu
num_epochs = 50  # Zwiększenie liczby epok

for epoch in range(num_epochs):
    for sequences, targets in dataloader:
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generowanie prognoz
model.eval()
test_data = torch.tensor(generate_synthetic_data(seq_len, 1), dtype=torch.float32).squeeze(0)
test_data = test_data.unsqueeze(0)  # Upewnij się, że kształt wejścia to [1, seq_len, input_size]

predictions = model(test_data).detach().numpy()

# Wykres
plt.figure(figsize=(7, 3.5))
plt.plot(test_data[0, :, 0], label='Position', color='C0')
plt.plot(test_data[0, :, 1], label='Velocity', color='C1', linestyle='--')
plt.xlabel("Time")
plt.ylabel("Markov State")
plt.legend(["Position", "Velocity"], fontsize=16)
plt.title("RNN Simulation")

plt.tight_layout()
plt.savefig("rnn_bouncing_ball.png")
plt.show()
