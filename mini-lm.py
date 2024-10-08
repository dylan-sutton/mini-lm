import PyPDF2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

pdf_path = 'example.pdf'

# Extract text from the PDF file
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

raw_text = extract_text_from_pdf(pdf_path)

# Clean the input text by removing special characters
def clean_text(text):
    text = text.lower()
    unwanted_chars = [
        '\n', '\r', '\t', '_', '*', '"', "'", '“', '”', '‘', '’', '—', '-', '–', '―', '−', '•', '∙', '·', '…', '°', '′', '″',
        '(', ')', '[', ']', '{', '}', '<', '>', '?', '!', '.', ',', ':', ';', '/', '=', '+', '^', '%', '$', '#', '@', '&', '~', '`', '|', '\\'
    ]
    for char in unwanted_chars:
        text = text.replace(char, ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    text = text.strip()
    return text

cleaned_text = clean_text(raw_text)

# Tokenize the cleaned text into words
def tokenize(text):
    return text.split(' ')

tokens = tokenize(cleaned_text)
print(f"Total tokens: {len(tokens)}")

# Build the vocabulary mappings: word2idx and idx2word
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

word2idx, idx2word = build_vocab(tokens)
vocab_size = len(word2idx)
print(f"Vocabulary Size: {vocab_size}")

# Encode tokens into their respective indices
def encode_tokens(tokens, word2idx):
    return [word2idx[word] for word in tokens]

encoded_tokens = encode_tokens(tokens, word2idx)

# Custom dataset to generate sequences of data for training
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + 1:index + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Hyperparameters
embedding_dim = 128
hidden_dim = 256
num_layers = 2
seq_length = 40
batch_size = 64
learning_rate = 0.001
epochs = 5

dataset = TextDataset(encoded_tokens, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# RNN-based language model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = output.reshape(-1, hidden_dim)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(num_layers, batch_size, hidden_dim)

model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()

# Training loop
for epoch in range(epochs):
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        batch_size_actual = inputs.size(0)
        if batch_size_actual != batch_size:
            hidden = hidden[:, :batch_size_actual, :].contiguous()

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        hidden = hidden.detach()

        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    average_loss = total_loss / len(dataloader)
    elapsed_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{epochs}] completed in {elapsed_time:.2f}s, Average Loss: {average_loss:.4f}")

# Save the trained model to disk
torch.save(model.state_dict(), "language_model.pth")
print("Model saved as 'language_model.pth'")

# Generate text using the trained model
def generate_text(model, start_text, num_words, word2idx, idx2word):
    model.eval()
    words = start_text.lower().split()
    hidden = model.init_hidden(1)

    for _ in range(num_words):
        x = torch.tensor([[word2idx.get(words[-1], 0)]], dtype=torch.long)
        output, hidden = model(x, hidden)
        probs = torch.softmax(output, dim=1).data
        word_id = torch.multinomial(probs, num_samples=1).item()
        words.append(idx2word[word_id])

    return ' '.join(words)

start_text = "the"
generated_text = generate_text(model, start_text, num_words=23, word2idx=word2idx, idx2word=idx2word)
print("Generated Text:")
print(generated_text)