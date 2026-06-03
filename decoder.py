import torch
import torch.nn as nn

class ClassicalDecoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, lstm_hidden_size=64, output_dim=2):
        super(ClassicalDecoder, self).__init__()

        # Input Layer (RC Embedding)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Recurrent Layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Readout Layer
        self.readout = nn.Linear(lstm_hidden_size, output_dim)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, input_dim)

        # Embed the binary quantum output into a continuous latent space
        embedded = self.relu(self.embedding(x))

        # Pass through LSTM
        # lstm_out is (batch_size, seq_len, lstm_hidden_size)
        lstm_out, _ = self.lstm(embedded)

        # Reconstruct the 2D process
        # Output shape is (batch_size, seq_len, output_dim)
        out = self.readout(lstm_out)

        return out

if __name__ == "__main__":
    model = ClassicalDecoder()
    dummy_input = torch.zeros((2, 10, 6))
    out = model(dummy_input)
    print("Decoder Input Shape:", dummy_input.shape)
    print("Decoder Output Shape:", out.shape)
