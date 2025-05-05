import datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL_MAPPING = {
    'MACHINE_GENERATED':1, 'HUMAN_GENERATED':0, 'MACHINE_REFINED':2, #'MACHINE_GENERATED_ADVERSARIAL':3
}
NUM_CLASSES = len(LABEL_MAPPING) 

encoded_dataset = datasets.load_from_disk('mbert_base_encoded_ternary')

print(f"Filtered train dataset len: {len(encoded_dataset['train'])}")
print(f"Filtered validation dataset len: {len(encoded_dataset['validation'])}")

from sentence_transformers import models, SentenceTransformer, losses
from transformers import AutoModel
text_model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')



TEXT_EMBEDDING_DIM = 768

def compute_class_weights(hf_dataset, label_mapping):
    label_ids = [label.item() for label in hf_dataset['labels']]
    label_counts = Counter(label_ids)
    print("Label counts:", label_counts)
    num_classes = len(label_mapping)
    total = sum(label_counts.values())
    if not label_counts or total == 0:
        print("Warning: No labels found or empty dataset. Using equal weights.")
        return torch.ones(num_classes, dtype=torch.float)
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count == 0:
            print(f"Warning: Class {i} not found in dataset. Assigning weight 1.0")
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * count))

    return torch.FloatTensor(weights)


print("Computing class weights...")
class_weights_tensor = compute_class_weights(encoded_dataset['train'], LABEL_MAPPING)
print("Class weights:", class_weights_tensor)


class TextDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        y = item['labels'].clone().detach().long()
        return {
            "input_ids": item['input_ids']['input_ids'],
            "attention_mask": item['input_ids']['attention_mask'],
            "labels": y,
        }

import torch
import torch.nn.functional as F


class DroidTextOnly(nn.Module):
    def __init__(self, text_encoder, projection_dim=128, num_classes=NUM_CLASSES, class_weights=None):
        super().__init__()
        self.text_encoder = text_encoder
        self.num_classes = num_classes

        text_output_dim = TEXT_EMBEDDING_DIM

        self.text_projection = nn.Linear(text_output_dim, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.class_weights = class_weights

    def forward(self, labels=None, input_ids=None, attention_mask=None):
        actual_labels = labels

       
        sentence_embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sentence_embeddings = sentence_embeddings.mean(dim=1)
       
        projected_text = F.relu(self.text_projection(sentence_embeddings))


        logits = self.classifier(projected_text)

        loss = None
        cross_entropy_loss = None

        if actual_labels is not None:
            loss_fct_ce = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
            cross_entropy_loss = loss_fct_ce(logits.view(-1, self.num_classes), actual_labels.view(-1))
            
            loss = cross_entropy_loss
        output = {"logits": logits, "fused_embedding": projected_text} # Also return embedding
        if loss is not None:
            output["loss"] = loss
        if cross_entropy_loss is not None:
             output["cross_entropy_loss"] = cross_entropy_loss
        return output


def fusion_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]


    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
       
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            labels = inputs["labels"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.get("loss")
            logits = outputs["logits"]

        logits = logits.detach()
        labels = labels.detach()

        if loss is None and not prediction_loss_only:
            loss_fct = nn.CrossEntropyLoss(weight=model.class_weights.to(logits.device) if model.class_weights is not None else None)
            loss = loss_fct(logits.view(-1, model.num_classes), labels.view(-1))

        return (loss, logits, labels)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return PyGDataLoader(
             self.train_dataset,
             batch_size=self.args.train_batch_size,
             shuffle=True,
             num_workers=self.args.dataloader_num_workers,
             collate_fn=fusion_collate_fn,
             pin_memory=self.args.dataloader_pin_memory 
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        return PyGDataLoader(
             eval_dataset,
             batch_size=self.args.eval_batch_size,
             shuffle=False,
             num_workers=self.args.dataloader_num_workers,
             collate_fn=fusion_collate_fn,
             pin_memory=self.args.dataloader_pin_memory
        )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

print("Creating Datasets...")
try:
     train_dataset = TextDataset(encoded_dataset['train'], )
     print(train_dataset[0])
     val_dataset = TextDataset(encoded_dataset['validation'], )
except ValueError as e:
     print(f"Error creating GMicrosoftraphDataset: {e}")
     print("Please check graph data integrity (nodes, edges, indices).")
     exit()
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


print("Instantiating models...")

fusion_model = DroidTextOnly(
    text_encoder=text_model,
    num_classes=3,
    projection_dim=256, 
    class_weights=class_weights_tensor
)


print("Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir="text_only_ternary_lr1e-5_modernbert_base", 
    evaluation_strategy="steps",
    eval_steps=250, 
    save_strategy="steps",
    save_steps=250,
    logging_dir="text_only_ternary_lr1e-5_modernbert_base",  
    logging_steps=50,
    learning_rate=1e-5, 
    per_device_train_batch_size=40,  
    per_device_eval_batch_size=128,  
    num_train_epochs=2,  
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_safetensors=False,
    report_to="wandb",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    fp16=torch.cuda.is_available(), 
)


print("Initializing Trainer...")
trainer = CustomTrainer(
    model=fusion_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
trainer.save_model('text_only_ternary_lr1e-5_modernbert_base/best_model')
