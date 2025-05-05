import datasets
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import numpy as np
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL_MAPPING = {
    'MACHINE_GENERATED':1, 'HUMAN_GENERATED':0,#'MACHINE_REFINED':2,#'MACHINE_GENERATED_ADVERSARIAL':3
}
NUM_CLASSES = len(LABEL_MAPPING)

train_dataset = datasets.load_from_disk('/l/users/daniil.orel/project_droid_graph_train_filtered_not_empty')
test_dataset = datasets.load_from_disk('/l/users/daniil.orel/project_droid_graph_test_filtered_not_empty')
val_dataset = datasets.load_from_disk('/l/users/daniil.orel/project_droid_graph_val_filtered+not_empty')
dataset = datasets.DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-large')

def preprocess_function(examples):
    tokenized = {}
    tokenized['input_ids'] = [tokenizer(i,
                         max_length=512, padding="max_length", truncation=True) for i in examples["Code"]]
    tokenized["labels"] = [LABEL_MAPPING[i ]for i in examples["Label"]]
    return tokenized



encoded_dataset = dataset.map(preprocess_function, batched=True, batch_size=2048)

encoded_dataset.set_format("torch", columns=["input_ids", "labels"])
encoded_dataset = encoded_dataset.remove_columns(["generated_AST_graph"])

encoded_dataset.save_to_disk("mbert_base_encoded_binary")
encoded_dataset = datasets.load_from_disk('mbert_large_encoded_binary')



from sentence_transformers import losses
from sentence_transformers.sampler import GroupByLabelBatchSampler
from transformers import AutoModel
text_model = AutoModel.from_pretrained('answerdotai/ModernBERT-large')


TEXT_EMBEDDING_DIM = 1024 # 768 for base, large is 1024

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
            # Assign a default high weight or handle as appropriate
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
        self.column_names = ['labels', 'input_ids']
    def __len__(self):
        return len(self.dataset)
    def get_label_list(self):
        return self.dataset['labels']
    def __getitem__(self, idx):
        if idx == 'labels':
            return self.get_label_list()
        
        item = self.dataset[idx]
        y = item['labels'].clone().detach().long()
        return {
            "input_ids": item['input_ids']['input_ids'],
            "attention_mask": item['input_ids']['attention_mask'],
            "labels": y,
        }

import torch
import torch.nn.functional as F


class TLModel(nn.Module):
    def __init__(self, text_encoder, projection_dim=128, num_classes=NUM_CLASSES, class_weights=None):
        super().__init__()
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        text_output_dim = TEXT_EMBEDDING_DIM
        self.additional_loss = losses.BatchHardSoftMarginTripletLoss(self.text_encoder)

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
        contrastive_loss = None

        if actual_labels is not None:
            loss_fct_ce = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
            cross_entropy_loss = loss_fct_ce(logits.view(-1, self.num_classes), actual_labels.view(-1))
            contrastive_loss = self.additional_loss.batch_hard_triplet_loss(embeddings=projected_text, labels=actual_labels)
            lambda_contrast = 0.1
            loss = cross_entropy_loss + lambda_contrast * contrastive_loss


        output = {"logits": logits, "fused_embedding": projected_text}
        if loss is not None:
            output["loss"] = loss
        if cross_entropy_loss is not None:
             output["cross_entropy_loss"] = cross_entropy_loss
        if contrastive_loss is not None:
             output["contrastive_loss"] = contrastive_loss

        return output

def fusion_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

from collections import defaultdict
from collections import Counter
import math
import random
class GroupByLabelBatchSamplerDebug(GroupByLabelBatchSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] = None,
        generator: torch.Generator = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        labels = [int(i) for i in self._determine_labels_to_use(dataset, valid_label_columns)]
        groups = defaultdict(list)
        for sample_idx, label in enumerate(labels):
            groups[label].append(sample_idx)

        self.groups = {
            label: indices[:len(indices) // 2 * 2]
            for label, indices in groups.items()
            if len(indices) >= 2
        }

        for label, indices in self.groups.items():
            print(f"Label {label}: {len(indices)} samples")

        self.label_list = list(self.groups.keys())

    def __iter__(self):
        if self.generator and self.seed:
            self.generator.manual_seed(self.seed + self.epoch)

        label_to_indices = {
            label: random.sample(indices, len(indices))
            for label, indices in self.groups.items()
        }

        num_labels = len(self.label_list)
        base_per_class = self.batch_size // num_labels
        remainder = self.batch_size % num_labels

        min_class_len = min(len(indices) for indices in label_to_indices.values())
        max_per_class = base_per_class + (1 if remainder > 0 else 0)
        max_batches = min_class_len // max_per_class

        for _ in range(max_batches):
            batch = []
            random.shuffle(self.label_list)

            for i, label in enumerate(self.label_list):
                take = base_per_class + (1 if i < remainder else 0)
                batch.extend(label_to_indices[label][:take])
                del label_to_indices[label][:take]

            yield batch

        if not self.drop_last:
            leftovers = [idx for indices in label_to_indices.values() for idx in indices]
            if len(leftovers) >= self.batch_size:
                yield leftovers[:self.batch_size]




class TLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = inputs['labels'].to(device)

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
        batch_sampler = GroupByLabelBatchSamplerDebug(self.train_dataset, self.args.train_batch_size,
                                                 drop_last=True, valid_label_columns=['labels'])
        return DataLoader(
             self.train_dataset,
            #  batch_size=self.args.train_batch_size,
             batch_sampler=batch_sampler,
             num_workers=self.args.dataloader_num_workers, 
             collate_fn=fusion_collate_fn,
             pin_memory=self.args.dataloader_pin_memory 
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        return DataLoader(
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
     print(train_dataset['labels'])
     print(train_dataset[0])
     val_dataset = TextDataset(encoded_dataset['validation'], )
except ValueError as e:
     exit()
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


print("Instantiating models...")

fusion_model = TLModel(
    text_encoder=text_model,
    num_classes=2,
    projection_dim=256,
    class_weights=class_weights_tensor 
)

print("Setting up Training Arguments...")
training_args = TrainingArguments(
    output_dir="binary_tl_model_lr1e-5_modernbert_large",
    evaluation_strategy="steps",
    eval_steps=250, 
    save_strategy="steps",
    save_steps=250,
    logging_dir="binary_tl_model_lr1e-5_modernbert_large", 
    logging_steps=50,
    learning_rate=1e-5, 
    per_device_train_batch_size=40,
    per_device_eval_batch_size=256, 
    num_train_epochs=3,
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
trainer = TLTrainer(
    model=fusion_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=fusion_collate_fn
)

print("Starting training...")
trainer.train()
trainer.save_model('binary_tl_model_lr1e-5_modernbert_large/best_model')
