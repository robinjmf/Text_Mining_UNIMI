import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, BertConfig

def get_encoded_output(input_text, model_checkpoint="distilbert-base-uncased", output_type="word"):
    # Load Distilbert Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: ", device)
    # model = AutoModel.from_pretrained(model_checkpoint).to(device)
    model = AutoModel.from_pretrained(model_checkpoint)



    config = BertConfig.from_pretrained(model_checkpoint)
    # Check the maximum sequence length
    max_sequence_length = config.max_position_embeddings
    # print("Maximum Sequence Length:", max_sequence_length)


    # Encode Input Text
    encoded_text = tokenizer(input_text, padding=True, truncation=True,max_length=50, add_special_tokens = True ,return_tensors='pt')  
    
    # Convert input to PyTorch tensor
    tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids[0])  # Extract tokens from the first sample
    # print(encoded_text, len(encoded_text.input_ids))
    # print(tokens)

    # Get encoded output vectors
    with torch.no_grad():
        output = model(**encoded_text)

    if output_type == "word":
        encoded_output = output.last_hidden_state  # Encoded representation for each word
    elif output_type == "sentence":
        # Extract the representation of the [CLS] token
        encoded_output = output.last_hidden_state[:, 0, :]  # Select the first token ([CLS]) from each sentence
    else:
        raise ValueError("Invalid output_type. Choose 'word' or 'sentence'.")

    return encoded_output


def get_encoded_output_(input_text, tokenizer, model, output_type="word"):
    # Encode Input Text
    encoded_text = tokenizer(input_text, padding=True, truncation=True, max_length=50, add_special_tokens=True, return_tensors='pt')  
    
    # Get encoded output vectors
    with torch.no_grad():
        output = model(**encoded_text)

    if output_type == "word":
        encoded_output = output.last_hidden_state  # Encoded representation for each word
    elif output_type == "sentence":
        # Extract the representation of the [CLS] token
        encoded_output = output.last_hidden_state[:, 0, :]  # Select the first token ([CLS]) from each sentence
    else:
        raise ValueError("Invalid output_type. Choose 'word' or 'sentence'.")

    return encoded_output






# Define your Siamese network architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Define the layers for each branch (embeddings1, embeddings2, embeddings3)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, emb1, emb2, emb3=None):
        # Implement the forward pass of your Siamese network
        # Combine the outputs from three branches and pass through additional layers if needed
        out1 = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(emb1)))))
        out2 = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(emb2)))))
        
        if emb3 is not None:
            for sample in emb3:
                if not torch.all(torch.isnan(sample)):
                    out3 = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(emb3)))))
                    return out1, out2, out3
            # If none of the samples passed the condition, return only out1 and out2
        return out1, out2

class MyDataset(Dataset):
    def __init__(self, embeddings1, embeddings2, labels, embeddings3=None):
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.embeddings3 = embeddings3
        self.labels = labels
        self.embedding_size = 768  # Assuming the size of each embedding tensor is 768

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb1 = self.process_embedding(self.embeddings1[idx])
        emb2 = self.process_embedding(self.embeddings2[idx])
        emb3 = self.process_embedding(self.embeddings3[idx]) if self.embeddings3 is not None else None
        label = self.labels[idx]
        return emb1, emb2, emb3, label

    def process_embedding(self, embedding):
        if embedding is None:
            return torch.full((self.embedding_size,), float('nan'))
        return embedding


# # Example usage:
# input_text = "The movie was not good"
# # Get encoded representations for each word
# word_embeddings = get_encoded_output(input_text, output_type="word")
# print("Word Embeddings:")
# print(word_embeddings)

# # Get a single encoded representation for the entire sentence
# sentence_embedding = get_encoded_output(input_text, output_type="sentence")
# print("Sentence Embedding:")
# print(sentence_embedding)
