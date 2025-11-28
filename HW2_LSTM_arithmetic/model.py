import torch

# --- Shared Hyperparameters ---
EMBED_DIM = 128
HIDDEN_DIM = 256

# --- Model Definition ---
class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx):
        super(CharRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn_layer1 = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.rnn_layer2 = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, batch_x, batch_x_lens):
        embedded = self.embedding(batch_x)
        # Clamp lengths to be at least 1, as pack_padded_sequence requires non-zero lengths
        clamped_lens = torch.clamp(batch_x_lens, min=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, clamped_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn_layer1(packed)
        packed_out, _ = self.rnn_layer2(packed_out)
        unpacked_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.linear(unpacked_out)
        return logits
    
    

    def generator(self, start_char, max_len, char_to_id, id_to_char):
        """
        This generator method is specific to evaluation/inference and is not used during training.
        It's kept here for convenience but could also live in the evaluation script.
        The implementation is moved to evaluate.py to keep the model definition clean.
        """
        # The implementation of the generator will now live in the evaluation script,
        # as it's not part of the core model's forward pass.
        pass