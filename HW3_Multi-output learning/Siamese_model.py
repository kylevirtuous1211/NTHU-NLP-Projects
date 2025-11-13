import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")

class SiameseMultiTaskModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. The shared-weight BERT model. This is the "Siamese" part.
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.hidden_size = self.bert.config.hidden_size
        
        # 2. Define the heads
        # The heads will operate on a *combination* of the two sentence vectors.
        # We will concatenate (vec_a, vec_b, |vec_a - vec_b|), so the
        # input size is 3 * hidden_size.
        
        # Input: [vec_a, vec_b, |vec_a - vec_b|]
        self.classification_head = torch.nn.Linear(self.hidden_size * 3, 3) 
        self.regression_head = torch.nn.Linear(self.hidden_size * 3, 1)

    def forward(self, 
                input_ids_a, attention_mask_a, token_type_ids_a,
                input_ids_b, attention_mask_b, token_type_ids_b,
                **kwargs):
        
        # 3. Pass Sentence A through BERT
        output_a = self.bert(input_ids=input_ids_a,
                             attention_mask=attention_mask_a,
                             token_type_ids=token_type_ids_a)
        # Use mean pooling for a better sentence vector
        vec_a = self.mean_pool(output_a.last_hidden_state, attention_mask_a)

        # 4. Pass Sentence B through the *SAME* BERT
        output_b = self.bert(input_ids=input_ids_b,
                             attention_mask=attention_mask_b,
                             token_type_ids=token_type_ids_b)
        vec_b = self.mean_pool(output_b.last_hidden_state, attention_mask_b)
        
        # 5. Combine the two vectors for the heads
        # This combination is a standard practice for Siamese sentence-pair tasks
        diff = torch.abs(vec_a - vec_b)
        combined_vec = torch.cat((vec_a, vec_b, diff), dim=1)
        
        # 6. Pass the combined vector to the heads
        classification_logits = self.classification_head(combined_vec)
        regression_score = self.regression_head(combined_vec)
        
        return classification_logits, regression_score

    # Helper function for mean pooling
    def mean_pool(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
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