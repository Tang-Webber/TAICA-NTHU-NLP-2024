# -*- coding: utf-8 -*-

import transformers as T
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam, RMSprop
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef, Accuracy, F1Score
device = "cuda:3" if torch.cuda.is_available() else "cpu"

# 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# Define the hyperparameters
lr = 1e-5
epochs = 6
train_batch_size = 8
validation_batch_size = 8
test_batch_size = 8

# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.

    # Preprocess data
    input_texts = [f"{item['premise']} [SEP] {item['hypothesis']}" for item in batch]
    # Tokenize
    input_encodings = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    # To tensors
    labels1 = torch.tensor([item['relatedness_score'] for item in batch])
    labels2 = torch.tensor([item['entailment_judgment'] for item in batch])

    return input_encodings, labels1, labels2

# TODO1-2: Define your DataLoader
dl_train = DataLoader(SemevalDataset(split="train"), batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
dl_validation = DataLoader(SemevalDataset(split="validation"), batch_size=validation_batch_size, collate_fn=collate_fn)
dl_test = DataLoader(SemevalDataset(split="test"), batch_size=test_batch_size, collate_fn=collate_fn)

model_name = "google-bert/bert-base-uncased"

# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define what modules you will use in the model
        self.bert = T.BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        # 3-Class Classification
        self.classifier = torch.nn.Linear(hidden_size, 3)

        # Regression
        self.regressor = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Extract BERT features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        # Apply linear layers
        relatedness = self.regressor(pooled_output)
        entailment = self.classifier(pooled_output)
        return relatedness, entailment

model = MultiLabelModel().to(device)
tokenizer = T.BertTokenizer.from_pretrained(model_name, cache_dir="./cache/")

#print(next(iter(dl_train)))

# TODO3: Define your optimizer and loss function

# TODO3-1: Define your Optimizer
optimizer = AdamW(model.parameters(), lr=lr)
# optimizer = RMSprop(model.parameters(), lr=lr)
# optimizer = Adam(model.parameters(), lr=lr)

# TODO3-2: Define your loss functions (you should have two)
loss_classifier = torch.nn.CrossEntropyLoss()
loss_regressor = torch.nn.MSELoss()

# scoring functions
spc = SpearmanCorrCoef()
acc = Accuracy(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3, average='macro')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

alpha = 0.9
smoothed_loss_reg = None
smoothed_loss_clf = None
threshold = 0.5

for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop

    # Train
    for batch in pbar:
        # clear gradient
        optimizer.zero_grad()

        # forward pass
        input_encodings, labels1, labels2 = batch
        input_ids = input_encodings.input_ids.to(device)
        attention_mask = input_encodings.attention_mask.to(device)
        token_type_ids = input_encodings.token_type_ids.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        relatedness, entailment = model(input_ids, attention_mask, token_type_ids)

        # compute loss
        loss_reg = loss_regressor(relatedness.squeeze(), labels1)
        loss_clf = loss_classifier(entailment, labels2)
        loss = loss_reg + loss_clf

        """
        (1) Directly Add
        loss_reg = loss_regressor(relatedness.squeeze(), labels1)
        loss_clf = loss_classifier(entailment, labels2)
        loss = loss_reg + loss_clf
        (2)
        # Prevent divided by zero error
        weight_reg = 1 / (loss_reg.item() + 1e-8)
        weight_clf = 1 / (loss_clf.item() + 1e-8)
        total_weight = weight_reg + weight_clf
        loss = (weight_reg / total_weight) * loss_reg + (weight_clf / total_weight) * loss_clf
        (3)
        loss_reg = loss_regressor(relatedness.squeeze(), labels1)
        loss_clf = loss_classifier(entailment, labels2)
        scaled_loss_reg = task_weights[0] * loss_reg
        scaled_loss_clf = task_weights[1] * loss_clf
        loss = scaled_loss_reg + scaled_loss_clf

        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        grads_norms = [torch.norm(g.detach(), p=2) for g in grads if g is not None]
        norm_reg = grads_norms[0] / grads_norms[0].item()
        norm_clf = grads_norms[1] / grads_norms[1].item()

        target_grad = torch.tensor([1.0, 1.0], device=device)
        lambda_reg = task_weights[0].data * (norm_reg / target_grad[0])
        lambda_clf = task_weights[1].data * (norm_clf / target_grad[1])
        new_task_weights = torch.tensor([lambda_reg, lambda_clf], device=device)
        task_weights.data = new_task_weights / new_task_weights.sum()
        (4)
        loss_reg = loss_regressor(relatedness.squeeze(), labels1)
        loss_clf = loss_classifier(entailment, labels2)
        if ep % 2 == 0:
            loss = loss_reg
        else:
            loss = loss_clf
        (5)
        loss_reg = loss_regressor(relatedness.squeeze(), labels1)
        loss_clf = loss_classifier(entailment, labels2)

        if smoothed_loss_reg is None:
            smoothed_loss_reg = loss_reg
            smoothed_loss_clf = loss_clf
        else:
            #smoothed_loss = alpha * loss.item() + (1 - alpha) * smoothed_loss
            smoothed_loss_reg = (1 - alpha) * loss_reg + alpha * smoothed_loss_reg.detach()
            smoothed_loss_clf = (1 - alpha) * loss_clf + alpha * smoothed_loss_clf.detach()

        loss = smoothed_loss_reg + smoothed_loss_clf
        (6)
        loss_reg = loss_regressor(relatedness.squeeze(), labels1)
        loss_clf = loss_classifier(entailment, labels2)

        loss = loss_reg * 0.6 + loss_clf * 0.4
        """

        # back-propagation
        loss.backward()

        # model optimization
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()


    # TODO5: Write the evaluation loop
    # Evaluate
    with torch.no_grad():
        relatedness_preds = []
        entailment_preds = []
        relatedness_labels = []
        entailment_labels = []

        #incorrect_samples = []

        for batch in pbar:
            input_encodings, labels1, labels2 = batch
            input_ids = input_encodings.input_ids.to(device)
            attention_mask = input_encodings.attention_mask.to(device)
            token_type_ids = input_encodings.token_type_ids.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            rel, ent = model(input_ids, attention_mask, token_type_ids)

            relatedness_preds.extend(rel.squeeze().tolist())
            entailment_preds.extend(ent.argmax(dim=1).tolist())
            relatedness_labels.extend(labels1.tolist())
            entailment_labels.extend(labels2.tolist())

        spc_score = spc(torch.tensor(relatedness_preds), torch.tensor(relatedness_labels))
        acc_score = acc(torch.tensor(entailment_preds), torch.tensor(entailment_labels))
        f1_score = f1(torch.tensor(entailment_preds), torch.tensor(entailment_labels))

        # Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)
        print(f"Spearman Corr: {spc_score:.3f}")
        print(f"Accuracy: {acc_score:.3f}")
        print(f"F1 Score: {f1_score:.3f}")

    torch.save(model, f'./saved_models/ep{ep}.ckpt')

"""For test set predictions, you can write perform evaluation simlar to #TODO5."""

pbar = tqdm(dl_test)
pbar.set_description(f"Test set evaluation")
model.eval()
with torch.no_grad():
    relatedness_preds = []
    entailment_preds = []
    relatedness_labels = []
    entailment_labels = []
    for batch in pbar:
        input_encodings, labels1, labels2 = batch
        input_ids = input_encodings.input_ids.to(device)
        attention_mask = input_encodings.attention_mask.to(device)
        token_type_ids = input_encodings.token_type_ids.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        rel, ent = model(input_ids, attention_mask, token_type_ids)

        relatedness_preds.extend(rel.squeeze().tolist())
        entailment_preds.extend(ent.argmax(dim=1).tolist())
        relatedness_labels.extend(labels1.tolist())
        entailment_labels.extend(labels2.tolist())

    spc_score = spc(torch.tensor(relatedness_preds), torch.tensor(relatedness_labels))
    acc_score = acc(torch.tensor(entailment_preds), torch.tensor(entailment_labels))
    f1_score = f1(torch.tensor(entailment_preds), torch.tensor(entailment_labels))

    print(f"Test set - Spearman Corr: {spc_score:.3f} | Accuracy: {acc_score:.3f} | F1 Score: {f1_score:.3f}")
    
"""
from collections import defaultdict
import re
from tqdm import tqdm
import random

pbar = tqdm(dl_test)
pbar.set_description("Test set evaluation")
model.eval()

error_log = defaultdict(list)
no_count = 0
total_error_count = 0

with torch.no_grad():
    relatedness_preds = []
    entailment_preds = []
    relatedness_labels = []
    entailment_labels = []

    for idx, batch in enumerate(pbar):
        input_encodings, labels1, labels2 = batch
        input_ids = input_encodings.input_ids.to(device)
        attention_mask = input_encodings.attention_mask.to(device)
        token_type_ids = input_encodings.token_type_ids.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        rel, ent = model(input_ids, attention_mask, token_type_ids)

        relatedness_preds_batch = rel.squeeze().tolist()
        entailment_preds_batch = ent.argmax(dim=1).tolist()
        relatedness_labels_batch = labels1.tolist()
        entailment_labels_batch = labels2.tolist()

        relatedness_preds.extend(relatedness_preds_batch)
        entailment_preds.extend(entailment_preds_batch)
        relatedness_labels.extend(relatedness_labels_batch)
        entailment_labels.extend(entailment_labels_batch)

        for i in range(len(relatedness_labels_batch)):
            if (round(relatedness_preds_batch[i]) != relatedness_labels_batch[i]) or \
               (entailment_preds_batch[i] != entailment_labels_batch[i]):
                total_error_count += 1
                input_ids_list = input_encodings.input_ids[i].tolist()
                decoded_text = tokenizer.decode(input_ids_list, skip_special_tokens=True)

                if re.search(r'\bno\b', decoded_text, re.IGNORECASE):
                    no_count += 1

                error_log["batch_index"].append(idx)
                error_log["data_index"].append(i)
                error_log["input_text"].append(decoded_text)
                error_log["true_relatedness"].append(relatedness_labels_batch[i])
                error_log["pred_relatedness"].append(relatedness_preds_batch[i])
                error_log["true_entailment"].append(entailment_labels_batch[i])
                error_log["pred_entailment"].append(entailment_preds_batch[i])

    spc_score = spc(torch.tensor(relatedness_preds), torch.tensor(relatedness_labels))
    acc_score = acc(torch.tensor(entailment_preds), torch.tensor(entailment_labels))
    f1_score = f1(torch.tensor(entailment_preds), torch.tensor(entailment_labels))

    print(f"Test set - Spearman Corr: {spc_score:.3f} | Accuracy: {acc_score:.3f} | F1 Score: {f1_score:.3f}")

print(f"Total errors logged: {total_error_count}")
print(f"Errors containing 'no': {no_count}/{total_error_count} ({(no_count / total_error_count) * 100:.2f}%)")

num_to_print = min(30, len(error_log['batch_index']))
random_indices = random.sample(range(len(error_log['batch_index'])), num_to_print)

for idx in random_indices:
    print(f"Error {idx + 1}:")
    print(f"  Batch Index: {error_log['batch_index'][idx]}")
    print(f"  Data Index: {error_log['data_index'][idx]}")
    print(f"  Input Text: {error_log['input_text'][idx]}")
    print(f"  True Relatedness: {error_log['true_relatedness'][idx]}, Predicted: {error_log['pred_relatedness'][idx]:.2f}")
    print(f"  True Entailment: {error_log['true_entailment'][idx]}, Predicted: {error_log['pred_entailment'][idx]}")

"""



















