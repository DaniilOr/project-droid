import datasets

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
 
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from transformers import Trainer
from torch.nn import CrossEntropyLoss

LABEL_MAPPING = {
    'MACHINE_GENERATED':1, 'HUMAN_GENERATED':0, #'MACHINE_REFINED':2, 'MACHINE_GENERATED_ADVERSARIAL':3
}

encoded_dataset = datasets.load_from_disk("encoded_data_binary")
print(encoded_dataset)
print(f"dataset len: {len(encoded_dataset['train'])}")

from unixcoder import UniXcoder

model = UniXcoder("microsoft/unixcoder-base")
import joblib # <-- Import joblib
try:
    VECTORIZER_PATH = "hashing_vectorizer_128_ast_node_texts.joblib" # Path to your saved vectorizer
    vectorizer = joblib.load(VECTORIZER_PATH)
    N_FEATURES = vectorizer.n_features # Get the number of features from the loaded vectorizer
    print(f"Vectorizer loaded successfully. n_features = {N_FEATURES}")
except FileNotFoundError:
    print(f"Error: Vectorizer file not found at {VECTORIZER_PATH}")
    print("Please ensure the vectorizer has been created and saved using the previous script.")
    exit()


import torch
import numpy as np
from collections import Counter

def compute_class_weights(hf_dataset, label_mapping):
    """
    Compute class weights from a HuggingFace dataset.
    
    Args:
        hf_dataset: a HuggingFace dataset split (e.g. train_dataset)
        label_mapping: dict mapping label names to integers
    
    Returns:
        torch.FloatTensor with class weights
    """
    label_ids = [label.item() for label in hf_dataset['labels']]
    label_counts = Counter(label_ids)
    print(label_counts)
    num_classes = len(label_mapping)
    total = sum(label_counts.values())
    weights = [total / (num_classes * label_counts[i]) for i in range(num_classes)]
    
    return torch.FloatTensor(weights)

class_weights_tensor = compute_class_weights(encoded_dataset['train'], LABEL_MAPPING)
print(class_weights_tensor)
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch

import json

class GraphDataset(Dataset):
    def __init__(self, hf_dataset, vectorizer, ):
        self.dataset = hf_dataset
        self.vectorizer = vectorizer
        self.n_features = self.vectorizer.n_features
        # Pre-compute zero feature vector for efficiency
        self.zero_feature = np.zeros(self.n_features, dtype=np.float16)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        ast_graph = item.get('generated_AST_graph')

        if ast_graph is None:
            raise ValueError(f"Graph at index {idx} is missing nodes, edges, or 'generated_AST_graph' itself.")

        nodes_data = ast_graph['nodes']
        num_nodes = len(nodes_data)

        node_features_list = [None] * num_nodes
        texts_to_vectorize = []
        indices_to_vectorize = []

        for i, node in enumerate(nodes_data):
            node_text = node.get("text", "")
            node_text = node_text.strip() if isinstance(node_text, str) else ""

            if not node_text:
                node_features_list[i] = self.zero_feature
            else:
                texts_to_vectorize.append(node_text)
                indices_to_vectorize.append(i)

        if texts_to_vectorize:
            sparse_features = self.vectorizer.transform(texts_to_vectorize)
            dense_features = sparse_features.toarray().astype(np.float16)

            for k, original_index in enumerate(indices_to_vectorize):
                node_features_list[original_index] = dense_features[k]

        if not node_features_list:
            x = torch.empty((0, self.n_features), dtype=torch.float)
        else:
            x = torch.from_numpy(np.stack(node_features_list)).float()

        if x.shape[0] != num_nodes or x.shape[1] != self.n_features:
            raise ValueError(f"Feature tensor x shape mismatch at index {idx}. "
                            f"Expected: [{num_nodes}, {self.n_features}], Got: {x.shape}")

        edge_index = torch.as_tensor(ast_graph['edges'], dtype=torch.long).t().contiguous()
        y = item['labels']

        graph_data = Data(x=x, edge_index=edge_index, y=y)

        return {
            "graph_data": graph_data,
            "input_ids": item["input_ids"],  # Text-based representation
            "labels": item["labels"]        
        }



class GCNForGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim) 
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
      
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
      
        x = global_mean_pool(x, batch)
        return x

TEXT_EMBEDDING_DIM = 768

class FusionModel(nn.Module):
    def __init__(self, gnn_encoder, text_encoder, projection_dim=128, num_classes=4):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.text_encoder = text_encoder

        self.gnn_projection = nn.Linear(128, projection_dim)
        self.text_projection = nn.Linear(TEXT_EMBEDDING_DIM, projection_dim) # Use determined dim

        self.classifier = nn.Linear(projection_dim * 2, num_classes)

    def forward(self, graph_data=None, input_ids=None, attention_mask=None, labels=None):
        graph_embedding = self.gnn_encoder(graph_data) # Shape: [batch_size, gnn_hidden_dim]

        outputs = self.text_encoder(input_ids)
        
        sentence_embeddings = outputs[1]

        projected_graph = F.relu(self.gnn_projection(graph_embedding))
        projected_text = F.relu(self.text_projection(sentence_embeddings))

        fused_embedding = torch.cat((projected_graph, projected_text), dim=-1)

        logits = self.classifier(fused_embedding)

        output = {"logits": logits}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, graph_data.y)
            output["loss"] = loss

        return output     

train_dataset = GraphDataset(encoded_dataset['train'], vectorizer,)
val_dataset = GraphDataset(encoded_dataset['validation'], vectorizer, )
test_dataset = GraphDataset(encoded_dataset['test'], vectorizer, )


from torch_geometric.loader import DataLoader as PyGDataLoader
 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def fusion_collate_fn(batch):
    """
    Collate function that batches both PyG graph data and text input tensors.
    
    Args:
        batch: A list of dicts, where each dict has:
            - 'graph_data': a torch_geometric.data.Data object
            - 'input_ids': a torch.Tensor
            - 'labels': a torch.Tensor

    Returns:
        A dict with:
            - 'graph_batch': a PyG Batch object
            - 'input_ids': a batched tensor of input_ids
            - 'labels': a batched tensor of labels
    """
    graph_data_list = [item['graph_data'] for item in batch]
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    graph_batch = Batch.from_data_list(graph_data_list)
    input_ids = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)

    return {
        'graph_batch': graph_batch,
        'input_ids': input_ids,
        'labels': labels
    }

class FusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        graph_data = inputs["graph_data"].to(device)
        input_ids = inputs["input_ids"].to(device)
        labels = graph_data.y # Labels are part of the graph batch
        outputs = model(
            graph_data=graph_data,
            input_ids=input_ids,
            labels=labels
        )
        loss = outputs["loss"]
        logits = outputs["logits"]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            # Put all tensor inputs on the correct device
            device = next(model.parameters()).device
            graph_data = inputs["graph_data"].to(device)
            input_ids = inputs["input_ids"].to(device)
            labels = graph_data.y 
            outputs = model(
                graph_data=graph_data,
                input_ids=input_ids,
                labels=labels
            )
            loss = outputs.get("loss", None)
            logits = outputs["logits"]

        logits = logits.detach()
        labels = labels.detach()

        return (loss, logits, labels)

    def get_train_dataloader(self):
        return DataLoader(
             self.train_dataset,
             batch_size=self.args.train_batch_size,
             shuffle=True,
            num_workers=32,
             collate_fn=fusion_collate_fn         )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader( 
             eval_dataset,
             batch_size=self.args.eval_batch_size,
             shuffle=False,
             num_workers=32,
             collate_fn=fusion_collate_fn 
        )
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

 

 

from transformers import TrainingArguments

gnn = GCNForGraphEncoder(input_dim=256,
                                   hidden_dim=128, ) 
model = FusionModel(gnn_encoder=gnn, text_encoder=model, num_classes=2) 


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="fusion_binary_ood_atcoder",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=250,
    logging_dir="fusion_binary_ood_atcoders",
    logging_steps=250,
    learning_rate=1e-4,
    per_device_train_batch_size=72,
    per_device_eval_batch_size=272,
    max_steps=5_000,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_safetensors=False,
    report_to="wandb", 
)

 

trainer = FusionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=fusion_collate_fn, 
    compute_metrics=compute_metrics,
)

trainer.train()
