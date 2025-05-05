import os
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
from transformers import AutoModel

LABEL_MAPPING = {
    'MACHINE_GENERATED':1, 'HUMAN_GENERATED':0,'MACHINE_REFINED':2,#'MACHINE_GENERATED_ADVERSARIAL':3
}
NUM_CLASSES = len(LABEL_MAPPING)


PRETRAINED_MODEL_CHECKPOINT_DIR = "best_model/pytorch_model.bin" 
HARD_CASE_PERCENTAGE = 0.10 
RETRAINING_OUTPUT_DIR = "hard_samples_mining_retraining_modernbert_base_ternary"
RETRAINING_LOGGING_DIR = os.path.join(RETRAINING_OUTPUT_DIR, "logs")
RETRAINING_EPOCHS = 2
RETRAINING_LEARNING_RATE = 1e-5 
RETRAINING_BATCH_SIZE = 40
try:
    encoded_dataset = datasets.load_from_disk('mbert_base_encoded_ternary')
    print(f"Original train dataset len: {len(encoded_dataset['train'])}")
    print(f"Original validation dataset len: {len(encoded_dataset['validation'])}")
except Exception as e:
    print(f"Error loading dataset: {e}", exc_info=True)
    exit()

print("Loading Text model...")
try:
    text_model = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
    TEXT_EMBEDDING_DIM = 768
    print(f"Loaded ModernBERT-base with embedding dimension: {TEXT_EMBEDDING_DIM}")
except Exception as e:
    print(f"Error loading text model: {e}", exc_info=True)
    exit()

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
from sentence_transformers import losses
from sentence_transformers.sampler import GroupByLabelBatchSampler

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

    def get_test_dataloader(self, test_dataset):
        if test_dataset is None:
             raise ValueError("Trainer: prediction requires a test_dataset.")
        return DataLoader( 
             test_dataset,
             batch_size=self.args.eval_batch_size, 
             shuffle=False, 
             num_workers=self.args.dataloader_num_workers,
             collate_fn=fusion_collate_fn, 
             pin_memory=self.args.dataloader_pin_memory
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()

    predictions = np.argmax(logits, axis=-1)
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
    except Exception as e:
        print(f"Error computing metrics: {e}", exc_info=True)
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


print("Creating initial TextDatasets...")
try:
    full_train_text_dataset = TextDataset(encoded_dataset['train'])
    val_text_dataset = TextDataset(encoded_dataset['validation']) 
    print(f"Original train TextDataset size: {len(full_train_text_dataset)}")
    print(f"Validation TextDataset size: {len(val_text_dataset)}")
except Exception as e:
    print(f"Error creating TextDataset: {e}", exc_info=True)
    exit()

print("Instantiating TLModel for prediction and loading pretrained weights...")
try:
    tl_model_for_prediction = TLModel(
        text_encoder=text_model, 
        num_classes=NUM_CLASSES, 
        projection_dim=256,
        class_weights=class_weights_tensor 
    )
    state_dict = torch.load(PRETRAINED_MODEL_CHECKPOINT_DIR)
    tl_model_for_prediction.load_state_dict(state_dict)
    print(f"Successfully loaded state dict from {PRETRAINED_MODEL_CHECKPOINT_DIR}")
except FileNotFoundError:
    print(f"Pretrained model checkpoint not found at {PRETRAINED_MODEL_CHECKPOINT_DIR}. Please set the correct path.")
    exit()
except Exception as e:
    print(f"Error loading pretrained model or state dict: {e}", exc_info=True)
    exit()


predict_args = TrainingArguments(
    output_dir="temp_prediction_output",
    per_device_eval_batch_size=256,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_dir=None,
    evaluation_strategy="no",
    save_strategy="no",
)

prediction_trainer = TLTrainer(
    model=tl_model_for_prediction, 
    args=predict_args,
    compute_metrics=None, 
    data_collator=fusion_collate_fn, 
)

print("Running prediction on the full training set to identify hard cases...")
try:
    prediction_output = prediction_trainer.predict(test_dataset=full_train_text_dataset)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids

    if labels is not None and labels.ndim > 1:
        labels = labels.squeeze()

    if logits is None or labels is None:
        print("Prediction output did not contain logits or labels. Cannot identify hard cases.")
        exit()
    print(f"Predictions obtained. Logits shape: {logits.shape}, Labels shape: {labels.shape}")

except Exception as e:
    print(f"Error during prediction on training set: {e}", exc_info=True)
    exit()


print(f"Identifying hardest {HARD_CASE_PERCENTAGE*100}% of cases per (label, Language) and (label, Source) group...")

try:
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy() # Convert to numpy
    true_class_probabilities = probabilities[np.arange(len(labels)), labels]
except Exception as e:
    print(f"Error calculating true class probabilities: {e}", exc_info=True)
    exit()

original_train_hf_dataset = encoded_dataset['train']

languages = original_train_hf_dataset['Language']
sources = original_train_hf_dataset['Source']
import pandas as pd
try:
    df_data = {
        'index': np.arange(len(labels)), 
        'label': labels,
        'true_prob': true_class_probabilities,
        'language': languages,
        'source': sources
    }
    pred_df = pd.DataFrame(df_data)
    print(f"Created DataFrame with {len(pred_df)} entries for analysis.")
    print(f"DataFrame columns: {pred_df.columns.tolist()}")
    print(f"DataFrame head:\n{pred_df.head()}")

except Exception as e:
    print(f"Error creating analysis DataFrame: {e}", exc_info=True)
    exit()

print("--- Selecting hardest cases based on (Label, Language) grouping ---")
hard_indices_lang_set = set()
grouping_cols_lang = ['label', 'language']

pred_df_lang = pred_df.copy()
if pred_df_lang[grouping_cols_lang].isnull().any().any():
    nan_counts_lang = pred_df_lang[grouping_cols_lang].isnull().sum()
    print(f"Found NaN values in Language grouping columns:\n{nan_counts_lang}. Rows with NaNs will be excluded from this grouping.")
    pred_df_lang = pred_df_lang.dropna(subset=grouping_cols_lang).copy() 

grouped_lang = pred_df_lang.groupby(grouping_cols_lang)
print(f"Number of unique (Label, Language) groups found: {len(grouped_lang)}")

for name, group in grouped_lang:
    group_size = len(group)
    if group_size == 0:
        continue
    n_hardest = max(1, math.ceil(group_size * HARD_CASE_PERCENTAGE))
    hardest_in_group = group.nsmallest(n_hardest, 'true_prob')
    hard_indices_lang_set.update(hardest_in_group['index'].tolist())

print(f"Selected {len(hard_indices_lang_set)} unique indices based on (Label, Language) grouping.")

print("--- Selecting hardest cases based on (Label, Source) grouping ---")
hard_indices_source_set = set()
grouping_cols_source = ['label', 'source']

pred_df_source = pred_df.copy()
if pred_df_source[grouping_cols_source].isnull().any().any():
    nan_counts_source = pred_df_source[grouping_cols_source].isnull().sum()
    print(f"Found NaN values in Source grouping columns:\n{nan_counts_source}. Rows with NaNs will be excluded from this grouping.")
    pred_df_source = pred_df_source.dropna(subset=grouping_cols_source).copy()

grouped_source = pred_df_source.groupby(grouping_cols_source)
print(f"Number of unique (Label, Source) groups found: {len(grouped_source)}")

for name, group in grouped_source:
    group_size = len(group)
    if group_size == 0:
        continue
    n_hardest = max(1, math.ceil(group_size * HARD_CASE_PERCENTAGE))

    hardest_in_group = group.nsmallest(n_hardest, 'true_prob')
    hard_indices_source_set.update(hardest_in_group['index'].tolist())

print(f"Selected {len(hard_indices_source_set)} unique indices based on (Label, Source) grouping.")

print("--- Combining indices from both groupings ---")
combined_hard_indices_set = hard_indices_lang_set.union(hard_indices_source_set)
hard_indices = np.array(sorted(list(combined_hard_indices_set))) # Convert set to sorted numpy array

if len(hard_indices) == 0:
    print("No hard examples were selected based on the combined criteria. No retraining will be performed.")
    exit()

print(f"Identified {len(hard_indices)} unique hard examples from the union of both grouping methods.")
print(f"Indices of hard examples (first 10): {hard_indices[:min(10, len(hard_indices))]}")

print("Creating new training dataset consisting of only the selected hard examples...")
try:
    hard_cases_hf_dataset = original_train_hf_dataset.select(hard_indices)
    hard_cases_text_dataset = TextDataset(hard_cases_hf_dataset)
    print(f"Hard cases training dataset size: {len(hard_cases_text_dataset)}")
    hard_case_labels = [item['labels'].item() for item in hard_cases_text_dataset]
    label_counts_hard_cases = Counter(hard_case_labels)
    print(f"Label distribution in hard cases dataset: {label_counts_hard_cases}")

except Exception as e:
    print(f"Error creating hard cases dataset: {e}", exc_info=True)
    exit()

print("Instantiating TLModel for retraining and loading pretrained weights...")
try:
    retraining_model = TLModel(
        text_encoder=text_model, 
        num_classes=NUM_CLASSES, 
        projection_dim=256, 
        class_weights=class_weights_tensor 
    )
    state_dict_retrain = torch.load(PRETRAINED_MODEL_CHECKPOINT_DIR, map_location='cpu') 
    retraining_model.load_state_dict(state_dict_retrain)
    print(f"Successfully loaded state dict for retraining from {PRETRAINED_MODEL_CHECKPOINT_DIR}")

    if torch.cuda.is_available():
        retraining_model.cuda()
        print("Retraining model moved to GPU.")
    else:
        print("CUDA not available. Retraining model will run on CPU.")

except FileNotFoundError:
    print(f"Pretrained model checkpoint not found at {PRETRAINED_MODEL_CHECKPOINT_DIR}. Please set the correct path.")
    exit()
except Exception as e:
    print(f"Error loading pretrained model for retraining or state dict: {e}", exc_info=True)
    exit()


print("Setting up Training Arguments for retraining...")


total_retraining_steps = len(hard_cases_text_dataset) // RETRAINING_BATCH_SIZE * RETRAINING_EPOCHS
eval_save_steps = max(1, total_retraining_steps // (RETRAINING_EPOCHS * 2)) 

retraining_args = TrainingArguments(
    output_dir=RETRAINING_OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=eval_save_steps,
    save_strategy="steps", 
    save_steps=eval_save_steps,
    logging_dir=RETRAINING_LOGGING_DIR,
    logging_steps=min(10, max(1, eval_save_steps // 2)),
    learning_rate=RETRAINING_LEARNING_RATE,
    per_device_train_batch_size=RETRAINING_BATCH_SIZE,
    per_device_eval_batch_size=256, 
    num_train_epochs=RETRAINING_EPOCHS,
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

print("Initializing Trainer for retraining...")
try:
    retraining_trainer = TLTrainer(
        model=retraining_model,
        args=retraining_args,
        train_dataset=hard_cases_text_dataset, 
        eval_dataset=val_text_dataset, 
        compute_metrics=compute_metrics, 
        data_collator=fusion_collate_fn, 
    )



except Exception as e:
    print(f"Error initializing retraining trainer: {e}", exc_info=True)
    exit()

import os

print("Starting retraining on hard cases...")
try:
    retraining_trainer.train()
    print("Retraining finished.")

    # Save the final best model
    final_model_save_path = os.path.join(RETRAINING_OUTPUT_DIR, 'best_model_retrained')
    print(f"Saving the best retrained model to {final_model_save_path}")
   
    retraining_trainer.save_model(final_model_save_path)
    print("Retrained model saved.")
except:
    pass
