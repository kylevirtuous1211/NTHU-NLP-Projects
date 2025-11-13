# %%
from transformers import ( BertTokenizer, BertModel, BertConfig, 
                          get_linear_schedule_with_warmup, RobertaTokenizer, 
                          AutoModel, GPT2Tokenizer)
from datasets import load_dataset
from evaluate import load
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
device = "cuda:1" if torch.cuda.is_available() else "cpu"
import argparse
import os
from pathlib import Path
#  You can install and import any other libraries if needed
# from Siamese_model import SiameseMultiTaskModel, collate_fn_siamese

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True) 
parser.add_argument(
    "--model_type", 
    type=str, 
    default="bert", 
    choices=["bert", "roberta", "gpt2"],
    help="Model type to train (bert, roberta, or gpt2)"
)

args = parser.parse_args()
checkpoint_dir = Path("checkpoints") / args.ckpt_dir
os.makedirs(checkpoint_dir, exist_ok=True)


# %%
# Some Chinese punctuations will be tokenized as [UNK], so we replace them with English ones
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

# %%
if args.model_type == "bert":
    model_checkpoint = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint, cache_dir="./cache/")
elif args.model_type == "roberta":
    model_checkpoint = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint, cache_dir="./cache/")
elif args.model_type == "gpt2":
    model_checkpoint = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, cache_dir="./cache/")
    # GPT-2 needs a pad token to be set
    tokenizer.pad_token = tokenizer.eos_token
    
# %%
class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "SemEvalWorkshop/sem_eval_2014_task_1",  # <-- THIS IS THE FIX
            trust_remote_code=True, split=split, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # Replace Chinese punctuations with English ones
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:100]
# print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")
# print(f"example: {data_sample}")
for ds in data_sample:
    if ds['entailment_judgment'] == 2:
        print(ds)

# %%
# Define the hyperparameters
# You can modify these values if needed
lr = 1e-4 # train from scratch 先試試看比較大的
weight_decay=0.01 # train from scratch 先試比較大
epochs = 10
train_batch_size = 32
validation_batch_size = 32

# %%
# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn_siamese(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    premise = []
    hypothesis = []
    relatedness_score = []
    entail_judge = []
    for data in batch:
        premise.append(data["premise"])
        hypothesis.append(data["hypothesis"])
        relatedness_score.append(data["relatedness_score"])
        entail_judge.append(data["entailment_judgment"])
    input_batch_a = tokenizer(premise, 
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=512 
                            )
    input_batch_b = tokenizer(hypothesis,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512 
                        )
    batch_dict = {}
    
    # Add premise tensors with '_a' suffix
    for key, value in input_batch_a.items():
        batch_dict[f"{key}_a"] = value
        
    # Add hypothesis tensors with '_b' suffix
    for key, value in input_batch_b.items():
        batch_dict[f"{key}_b"] = value

    # Add labels
    batch_dict["relatedness_score"] = torch.tensor(relatedness_score, dtype=torch.float)
    batch_dict["entailment_judgment"] = torch.tensor(entail_judge, dtype=torch.long)
    
    # Return the single, combined batch
    return batch_dict

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    premise = []
    hypothesis = []
    relatedness_score = []
    entail_judge = []
    for data in batch:
        premise.append(data["premise"])
        hypothesis.append(data["hypothesis"])
        relatedness_score.append(data["relatedness_score"])
        entail_judge.append(data["entailment_judgment"])
    input_batch = tokenizer(premise, 
                            hypothesis,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=512 
                            )
    input_batch["relatedness_score"] = torch.tensor(relatedness_score, dtype=torch.float)
    input_batch["entailment_judgment"] = torch.tensor(entail_judge, dtype=torch.long)
    
    return input_batch

# TODO1-2: Define your DataLoader
train_dataset = SemevalDataset(split="train")
val_dataset = SemevalDataset(split="validation")
test_dataset = SemevalDataset(split="test")
# dl_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
# dl_validation = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)
# dl_test = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)
# Siamese
dl_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn_siamese)
dl_validation = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn_siamese)
dl_test = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn_siamese)

# %%
# TODO2: Construct your model
class SiameseMultiTaskModel(torch.nn.Module):
    def __init__(self, model_checkpoint="google-bert/bert-base-uncased", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. The shared-weight BERT model. This is the "Siamese" part.
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.hidden_size = self.bert.config.hidden_size
        
        # 2. Define the heads
        # The heads will operate on a *combination* of the two sentence vectors.
        # We will concatenate (vec_a, vec_b, |vec_a - vec_b|), so the
        # input size is 3 * hidden_size.
        
        # Input: [vec_a, vec_b, |vec_a - vec_b|]
        self.classification_head = torch.nn.Linear(self.hidden_size * 3, 3) 
        self.regression_head = torch.nn.Linear(self.hidden_size * 3, 1)

    def forward(self, 
                input_ids_a, attention_mask_a, input_ids_b, 
                attention_mask_b,  token_type_ids_a=None, token_type_ids_b=None,
                **kwargs):
        
        # create input dictionaries for gpt2/Roberta/Bert
        inputs_a = {
            "input_ids": input_ids_a,
            "attention_mask": attention_mask_a
        }
        inputs_b = {
            "input_ids": input_ids_b,
            "attention_mask": attention_mask_b
        }
        
        # Only add token_type_ids if they are provided (i.e., for BERT)
        if token_type_ids_a is not None:
            inputs_a["token_type_ids"] = token_type_ids_a
        if token_type_ids_b is not None:
            inputs_b["token_type_ids"] = token_type_ids_b
            
        output_a = self.bert(**inputs_a)
        output_b = self.bert(**inputs_b)

        # 6. Handle different output types (BERT/RoBERTa vs. GPT-2)
        # BERT/RoBERTa output an object with .last_hidden_state
        if hasattr(output_a, 'last_hidden_state'):
            last_hidden_a = output_a.last_hidden_state
            last_hidden_b = output_b.last_hidden_state
        # GPT-2 outputs a tuple, where the first item is the last_hidden_state
        else:
            last_hidden_a = output_a[0]
            last_hidden_b = output_b[0]

        vec_a = self.mean_pool(last_hidden_a, attention_mask_a)
        vec_b = self.mean_pool(last_hidden_b, attention_mask_b)
        
        # 5. Combine the two vectors for the heads
        # This combination is a standard practice for Siamese sentence-pair tasks
        diff = torch.abs(vec_a - vec_b)
        combined_vec = torch.cat((vec_a, vec_b, diff), dim=1)
        
        # 6. Pass the combined vector to the heads
        classification_logits = self.classification_head(combined_vec)
        regression_score = self.regression_head(combined_vec)
        
        return classification_logits, regression_score
    
    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
        

# %%
# TODO3: Define your optimizer and loss function
# Cross Encoder
# model = MultiLabelModel().to(device)
# Siamese Encoder
model = SiameseMultiTaskModel(model_checkpoint=model_checkpoint).to(device)

# TODO3-1: Define your Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler
num_training_steps = len(dl_train) * epochs
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
# TODO3-2: Define your loss functions (you should have two)
# Write your code here
loss_classification = torch.nn.CrossEntropyLoss()
loss_regression = torch.nn.MSELoss()
regression_weight = 10
# scoring functions
psr = load("pearsonr")
acc = load("accuracy")

# %%
best_score = 0.0
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    for i, batch in enumerate(pbar):
        device_batch = {k:v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        # FIX: pop the labels
        label_cls = device_batch.pop("entailment_judgment")
        label_reg = device_batch.pop("relatedness_score")
        classification_logits, regression_score = model(**device_batch)
        # label_reg = device_batch["relatedness_score"]
        # label_cls = device_batch["entailment_judgment"]

        label_reg = label_reg.unsqueeze(1)
        loss_cls = loss_classification(classification_logits, label_cls)
        loss_reg = loss_regression(regression_score, label_reg)
        
        total_loss = loss_cls + loss_reg * regression_weight
        if i % 100 == 0: # Print every 100 batches
            print(f"Current LR: {optimizer.param_groups[0]['lr']}")
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({'loss': total_loss.item()})


    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    # Output all the evaluation scores (PearsonCorr, Accuracy)
    all_preds_cls = []
    all_labels_cls = []
    all_preds_reg = []
    all_labels_reg = []
    with torch.no_grad():
        for batch in pbar:
            device_batch = {k:v.to(device) for k, v in batch.items()}
            classification_logits, regression_score = model(**device_batch)
            label_cls = device_batch["entailment_judgment"]
            label_reg = device_batch["relatedness_score"]
            preds_cls = torch.argmax(classification_logits, dim=1)
            
            all_preds_cls.append(preds_cls.cpu())
            all_labels_cls.append(label_cls.cpu())
            all_preds_reg.append(regression_score.cpu())
            all_labels_reg.append(label_reg.cpu())
        
    all_preds_cls = torch.cat(all_preds_cls)
    all_labels_cls = torch.cat(all_labels_cls)
    all_preds_reg = torch.cat(all_preds_reg).squeeze() # Squeeze to 1D
    all_labels_reg = torch.cat(all_labels_reg)

    accuracy = acc.compute(predictions=all_preds_cls, references=all_labels_cls)
    pearson_corr = psr.compute(predictions=all_preds_reg, references=all_labels_reg)
    accuracy_value = accuracy['accuracy']
    pearson_value = pearson_corr['pearsonr']
            # print(f"F1 Score: {f1.compute()}")
    print(f"Epoch {ep}: validation accuracy: {accuracy_value}, pearson_value: {pearson_value} ")
        
    if pearson_value + accuracy_value > best_score:
        best_score = pearson_value + accuracy_value
        torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.ckpt')
        print(f"New best score: {best_score:.4f}. Model saved.")

# %%
# Cross Encoder approach
# model = MultiLabelModel().to(device)
# Siamese Model
model = SiameseMultiTaskModel(model_checkpoint=model_checkpoint).to(device)
model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.ckpt", weights_only=True))

# Test Loop
pbar = tqdm(dl_test, desc="Test")
model.eval()

# TODO6: Write the test loop
# Write your code here
# We have loaded the best model with the highest evaluation score for you
# Please implement the test loop to evaluate the model on the test dataset
# We will have 10% of the total score for the test accuracy and pearson correlation
all_preds_cls = []
all_labels_cls = []
all_preds_reg = []
all_labels_reg = []
with torch.no_grad():
    for batch in pbar:
        device_batch = {k:v.to(device) for k, v in batch.items()}
        classification_logits, regression_score = model(**device_batch)
        label_cls = device_batch["entailment_judgment"]
        label_reg = device_batch["relatedness_score"]
        preds_cls = torch.argmax(classification_logits, dim=1)
        
        all_preds_cls.append(preds_cls.cpu())
        all_labels_cls.append(label_cls.cpu())
        all_preds_reg.append(regression_score.cpu())
        all_labels_reg.append(label_reg.cpu())
    
all_preds_cls = torch.cat(all_preds_cls)
all_labels_cls = torch.cat(all_labels_cls)
all_preds_reg = torch.cat(all_preds_reg).squeeze() # Squeeze to 1D
all_labels_reg = torch.cat(all_labels_reg)

accuracy = acc.compute(predictions=all_preds_cls, references=all_labels_cls)
pearson_corr = psr.compute(predictions=all_preds_reg, references=all_labels_reg)

print(f"test accuracy: {accuracy}")
print(f"test pearson_corr: {pearson_corr}")

# start analyzing the wrong data points
import pandas as pd

premise = [d['premise'] for d in test_dataset]
hypothesis = [d['hypothesis'] for d in test_dataset]

df = pd.DataFrame({
    "all_preds_reg": all_preds_reg.cpu().numpy(), 
    "all_preds_cls": all_preds_cls.cpu().numpy(),
    "all_labels_cls": all_labels_cls.cpu().numpy(),
    "all_labels_reg": all_labels_reg.cpu().numpy(),
    "premise": premise,
    "entailment": hypothesis}
)

df_wrong_cls = df[df['all_preds_cls'] != df['all_labels_cls']]
wrong_cls_file = "bert_wrong_cls_best_model.csv"
print(f"writing to {wrong_cls_file}")
df_wrong_cls.to_csv(wrong_cls_file, index=False)

df['reg_error'] = abs(df['all_preds_reg'] - df['all_labels_reg'])
df_wrong_reg = df[df['reg_error'] >= 2].sort_values('reg_error', ascending=False)
wrong_reg_file = "bert_wrong_reg_best_model.csv"
print(f"writing to {wrong_reg_file}")
df_wrong_reg.to_csv(wrong_reg_file, index=False)
 