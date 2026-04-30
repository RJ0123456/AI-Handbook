import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		# nn.LSTM reads one sequence step by step and keeps a hidden state.
		# batch_first=True means input shape is (batch_size, seq_len, input_size).
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
		)

		# The last time step output is mapped to class scores.
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		batch_size = x.size(0)

		# h0: initial hidden state, c0: initial cell state.
		# Shape = (num_layers, batch_size, hidden_size)
		h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
		c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

		# lstm_out shape: (batch_size, seq_len, hidden_size)
		# hn shape: (num_layers, batch_size, hidden_size)
		lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

		# Use the output from the last time step as the sequence summary.
		last_output = lstm_out[:, -1, :]

		# Convert the sequence summary into class logits.
		logits = self.fc(last_output)
		return logits


if __name__ == "__main__":
	torch.manual_seed(42)

	# Each sample is a sequence with 4 time steps.
	# Each time step has 2 features.
	x_train = torch.tensor(
		[
			[[0.0, 0.0], [0.1, 0.2], [0.0, 0.1], [0.2, 0.1]],
			[[1.0, 1.1], [1.2, 0.9], [1.1, 1.0], [1.3, 1.2]],
			[[0.1, 0.0], [0.2, 0.1], [0.1, 0.2], [0.0, 0.1]],
			[[1.1, 1.0], [1.0, 1.2], [1.2, 1.1], [1.3, 1.0]],
		],
		dtype=torch.float32,
	)

	# Class 0 = small values, Class 1 = large values.
	y_train = torch.tensor([0, 1, 0, 1], dtype=torch.long)

	model = LSTMClassifier(
		input_size=2,
		hidden_size=8,
		num_layers=1,
		num_classes=2,
	)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	print("Input shape:", x_train.shape)
	print("Label shape:", y_train.shape)

	for epoch in range(1, 201):
		model.train()

		# Forward pass: model predicts class logits for each sequence.
		logits = model(x_train)
		loss = criterion(logits, y_train)

		# Backward pass: compute gradients and update weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch % 50 == 0:
			print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

	model.eval()
	with torch.no_grad():
		logits = model(x_train)
		predictions = torch.argmax(logits, dim=1)

	print("Predicted classes:", predictions.tolist())
	print("True classes     :", y_train.tolist())
