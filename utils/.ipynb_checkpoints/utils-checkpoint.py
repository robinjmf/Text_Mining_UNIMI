import torch
import torch.nn.functional as F 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models import MyDataset, get_encoded_output_
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

def chunk_text(text, chunk_size=492, chunk_overlap=20):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200B",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],)
    docs = text_splitter.create_documents([text])

    return docs
    


import os
import json
import pandas as pd
import logging

# Load Distilbert Tokenizer and Model
from transformers import AutoTokenizer, AutoModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_training_dataset(data_path, ct_json_dir):

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    with open(data_path) as json_file:
        data_ = json.load(json_file)

    uuid_list = list(data_.keys())
    statements = []
    labels = []
    dataset = []

    total_iterations = len(uuid_list)
    for i, uuid in enumerate(uuid_list):
        # Log the progress
        # Calculate progress percentage
        progress_percentage = (i + 1) / total_iterations * 100
        if progress_percentage % 2 == 0:
            logger.info("Processing %d/%d (%.2f%%) iterations", i + 1, total_iterations, progress_percentage)

        # Retrieve all statements and labels from the data
        encoded_statement = get_encoded_output_(data_[uuid]["Statement"], tokenizer, model, output_type="sentence")
        statements.append(encoded_statement)
        labels.append(data_[uuid]["Label"])

        primary_ctr_path = os.path.join(ct_json_dir, data_[uuid]["Primary_id"] + ".json")
        with open(primary_ctr_path) as json_file:
            primary_ctr = json.load(json_file)

        # Retrieve the full section from the primary trial
        primary_ctr_text = '\n'.join(primary_ctr[data_[uuid]["Section_id"]])
        chunks = chunk_text(primary_ctr_text)
        # computing the encoded output for each chunk of primary section
        chunk_encoded=[]
        for chunk in chunks:
            chunk_encoded.append(get_encoded_output_(str(chunk), tokenizer, model, output_type="sentence"))

        # Convert your list of tensors to a single tensor
        stacked_tensors = torch.stack(chunk_encoded)

        # Calculate the mean across all tensors
        avg_encoded_primary_section = torch.mean(stacked_tensors, dim=0)
        primary_section = avg_encoded_primary_section
        primary_statement_pair = (statements[i], primary_section)

        if data_[uuid]["Type"] == "Comparison":
            secondary_ctr_path = os.path.join(ct_json_dir, data_[uuid]["Secondary_id"] + ".json")
            with open(secondary_ctr_path) as json_file:
                secondary_ctr = json.load(json_file)
            secondary_section = secondary_ctr[data_[uuid]["Section_id"]]
            encoded_secondary_section = get_encoded_output_(secondary_section, tokenizer, model, output_type="sentence")

            secondary_statement_pair = (statements[i], encoded_secondary_section)

            dataset.append((primary_statement_pair, secondary_statement_pair))
        else:
            dataset.append((primary_statement_pair, None))

    # Create a pandas DataFrame
    df = pd.DataFrame(dataset, columns=["Primary_Statement", "Secondary_Statement"])
    df["Label"] = labels

    return df

# # Example usage
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("distilbert-base-uncased")

# train_path = "training_data/train.json"
# ct_json_dir = "training_data/CT json"
# training_dataset_df = prepare_training_dataset(train_path, ct_json_dir)



def contrastive_loss(output1, output2, label, margin=1.0):
    distance = F.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                      label * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss



def my_collate_fn(batch):
    # Check if the third element of any data in the batch is None
    has_none = any(data[2] is None for data in batch)

    # If any data in the batch has the third element as None
    if has_none:
        # Filter out None values from the batch while keeping other elements intact
        filtered_batch = [(data[0], data[1], data[3]) if data[2] is None else data for data in batch]
        return default_collate(filtered_batch)
    else:
        # Use the default collate function for the entire batch
        return default_collate(batch)
    




import torch.nn.functional as F

def load_data(dataset_df, batch_size=64, shuffle=False):
    # Fit and transform the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset_df["Label"])

    embeddings1 = dataset_df['Primary_Statement'].apply(lambda tensor: tensor[0][0]).tolist()
    embeddings2 = dataset_df['Primary_Statement'].apply(lambda tensor: tensor[1][0]).tolist()
    embeddings3 = dataset_df['Secondary_Statement'].apply(lambda tensor: tensor[1][0] if tensor is not None else None).tolist()
    labels = encoded_labels.tolist()

    # Define DataLoader for validation dataset
    dataset = MyDataset(embeddings1, embeddings2, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn)

    return dataloader, label_encoder




def contrastive_loss(output1, output2, label, margin=1.0):
    distance = F.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                      label * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss


