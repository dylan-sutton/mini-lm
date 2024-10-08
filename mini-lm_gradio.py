import gradio as gr
import PyPDF2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os

# Extract text from the PDF file
def extract_text_from_pdf(pdf_path):
    # Load the PDF and extract text from each page
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        if page.extract_text():
            text += page.extract_text()
    return text

# Clean the input text by removing special characters
def clean_text(text):
    # Convert all text to lowercase
    text = text.lower()
    # Define unwanted characters to remove
    unwanted_chars = [
        '\n', '\r', '\t', '_', '*', '"', "'", '“', '”', '‘', '’', '—', '-', '–', '―', '−', '•', '∙', '·', '…', '°', '′', '″',
        '(', ')', '[', ']', '{', '}', '<', '>', '?', '!', '.', ',', ':', ';', '/', '=', '+', '^', '%', '$', '#', '@', '&', '~', '`', '|', '\\'
    ]
    # Remove unwanted characters from the text
    for char in unwanted_chars:
        text = text.replace(char, ' ')
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    return text

# Tokenize the cleaned text into words
def tokenize(text):
    # Split the text by spaces to create tokens
    return text.split(' ')

# Build the vocabulary mappings: word2idx and idx2word
def build_vocab(tokens):
    # Create a sorted list of unique tokens to build the vocabulary
    vocab = sorted(set(tokens))
    # Create mapping from words to indices and vice versa
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# Encode tokens into their respective indices
def encode_tokens(tokens, word2idx):
    # Convert each token to its corresponding index using the word2idx dictionary
    return [word2idx[word] for word in tokens if word in word2idx]

# Custom dataset to generate sequences of data for training
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # The length is reduced by the sequence length to avoid out-of-bounds errors
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # Get the input sequence and the target sequence
        x = self.data[index:index + self.seq_length]
        y = self.data[index + 1:index + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# RNN-based language model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNLanguageModel, self).__init__()
        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN layer to process the sequences
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer to map the RNN output to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden):
        # Embed the input indices
        embedded = self.embedding(x)
        # Pass the embedded input through the RNN
        output, hidden = self.rnn(embedded, hidden)
        # Reshape output to pass it through the fully connected layer
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

# Load and train model
def train_model(pdf_path):
    # Extract and clean text from the uploaded PDF
    raw_text = extract_text_from_pdf(pdf_path.name)
    cleaned_text = clean_text(raw_text)
    # Tokenize the text and build vocabulary
    tokens = tokenize(cleaned_text)
    word2idx, idx2word = build_vocab(tokens)
    # Encode tokens into indices
    encoded_tokens = encode_tokens(tokens, word2idx)
    vocab_size = len(word2idx)

    # Hyperparameters for the model
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    seq_length = 40
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    # Create the dataset and data loader
    dataset = TextDataset(encoded_tokens, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the RNN language model, loss function, and optimizer
    model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    # Training loop
    for epoch in range(epochs):
        # Initialize the hidden state at the start of each epoch
        hidden = model.init_hidden(batch_size)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_size_actual = inputs.size(0)
            # Adjust the hidden state if the last batch is smaller
            hidden = hidden[:, :batch_size_actual, :].contiguous()

            # Zero the gradients, perform forward pass, calculate loss, and backpropagate
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            hidden = hidden.detach()  # Detach hidden state to avoid backpropagating through the entire history

            # Calculate the loss and update the model parameters
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()

    # Save the trained model
    model_path = "language_model.pth"
    torch.save(model.state_dict(), model_path)
    return "Model trained and saved as 'language_model.pth'", model_path, vocab_size, embedding_dim, hidden_dim, num_layers, word2idx, idx2word

# Generate text using the trained model
def generate_text(start_text, num_words, vocab_size, embedding_dim, hidden_dim, num_layers, word2idx, idx2word):
    # Load the trained model
    model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load("language_model.pth"))
    model.eval()
    words = start_text.lower().split()
    # Initialize hidden state for generating text
    hidden = model.init_hidden(1)

    for _ in range(num_words):
        # Get the index of the last word in the sequence
        x = torch.tensor([[word2idx.get(words[-1], 0)]], dtype=torch.long)
        # Perform a forward pass to get the output probabilities
        output, hidden = model(x, hidden)
        probs = torch.softmax(output, dim=1).data
        # Sample the next word based on the output probabilities
        word_id = torch.multinomial(probs, num_samples=1).item()
        words.append(idx2word[word_id])

    return ' '.join(words)

# Gradio interface
def gradio_train(pdf):
    # Train the model using the uploaded PDF
    result, model_path, vocab_size, embedding_dim, hidden_dim, num_layers, word2idx, idx2word = train_model(pdf)
    # Store the model parameters to use in the text generation step
    gradio_train.vocab_size = vocab_size
    gradio_train.embedding_dim = embedding_dim
    gradio_train.hidden_dim = hidden_dim
    gradio_train.num_layers = num_layers
    gradio_train.word2idx = word2idx
    gradio_train.idx2word = idx2word
    gradio_train.model_path = model_path
    return result

def gradio_generate(start_text, num_words):
    # Generate text using the trained model and user input
    return generate_text(start_text, int(num_words), gradio_train.vocab_size, gradio_train.embedding_dim, gradio_train.hidden_dim, gradio_train.num_layers, gradio_train.word2idx, gradio_train.idx2word)

# Function to download the trained model
def download_model():
    if hasattr(gradio_train, 'model_path') and os.path.exists(gradio_train.model_path):
        return gr.File.update(value=gradio_train.model_path, visible=True)
    else:
        return "Model not found. Please train the model first."

# Define the Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown("""
    # Mini Language Model Trainer
    This application lets you train a simple language model using text extracted from a PDF file. The trained model can then generate text based on user input. It is ideal for experimenting with basic natural language processing and text generation tasks.
    """)
    # Tab for training the model
    with gr.Tab("Train Model"):
        pdf_input = gr.File(label="Upload PDF")
        train_button = gr.Button("Train Model")
        train_output = gr.Textbox()
        train_button.click(fn=gradio_train, inputs=pdf_input, outputs=train_output)
        # Button to download the trained model
        download_button = gr.Button("Download Trained Model")
        download_file = gr.File(visible=False)
        download_button.click(fn=download_model, inputs=None, outputs=download_file)
    # Tab for generating text using the trained model
    with gr.Tab("Generate Text"):
        start_text_input = gr.Textbox(label="Start Text")
        num_words_input = gr.Number(label="Number of Words to Generate")
        generate_button = gr.Button("Generate Text")
        generate_output = gr.Textbox()
        generate_button.click(fn=gradio_generate, inputs=[start_text_input, num_words_input], outputs=generate_output)

# Launch the Gradio app
demo.launch()