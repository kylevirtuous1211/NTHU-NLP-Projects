# Multi-Output Learning for Sentence-Pair Tasks

This project trains and evaluates multi-task transformer models (BERT, RoBERTa, GPT-2) on two sentence-pair tasks simultaneously:
1.  **Relatedness Score (Regression):** Predicting a semantic relatedness score.
2.  **Entailment Judgement (Classification):** Classifying the relationship as Entailment, Contradiction, or Neutral.

The core architecture is a **Siamese (bi-encoder)** network, which creates independent representations for the premise and hypothesis before comparing them.

## 1. Setup

Clone the repository and install the required dependencies:

```
git clone [https://your-repo-url.git](https://your-repo-url.git)
cd your-repo-directory
pip install -r requirements.txt
```
## How to Train & Evaluate
The main script training_script.py will train the model, save the best-performing checkpoint, and then automatically run the test loop on the dl_test dataset.

All logs (both stdout and stderr) will be redirected to a .log file.

Arguments
* --model_type: The base model to use. (Choices: "bert", "roberta", "gpt2")
* --ckpt_dir: The name of the directory where the `best_model.ckpt` will be saved.

### Example Run
This command trains a BERT-base model for 10 epochs with a batch size of 32, a learning rate of 1e-4, and a regression weight of 10. (You have to updated hyperparameters in the code yourself)
```
nohup python training_script.py \
    --ckpt_dir 1111_Siamese_lr=1e-4_wd=0.01_e10_b32_with_scheduler_rw10 \
    --model_type bert \
    > logs/1111_Siamese_lr=1e-4_wd=0.01_e10_b32_with_scheduler_rw10.log 2>&1 &
```

## 3. Ablation Study (Single-Task Training)


To compare the multi-task model against single-task baselines, you can use `training_script_separately.py`.

This requires a manual code change to select which task to train on.

**Step 1**: Open `training_script_separately.py`. 
**Step 2**: Go to the loss calculation section (around line 302).

```
        # uncomment this if you only want relatedness regression task 
        total_loss = loss_reg * regression_weight
        # uncomment this if you only want entailment classification
        # total_loss = loss_cls
```
**Step 2**: Run the script. The example below trains a regression-only model.
```
nohup python training_script_separately.py \
    --ckpt_dir 1112_only_reg_Siamese_lr=1e-4_wd=0.01_e10_b32_with_scheduler_rw10 \
    --model_type bert \
    > logs/only_reg/1112_only_reg_Siamese_lr=1e-4_wd=0.01_e10_b32_with_scheduler_rw10.log 2>&1 &
```

## Key Architectural Notes
### Loss Weighting
Initial experiments showed the model learned the classification task easily but failed to learn the regression task (low Pearson score).

To fix this, we introduced regression_weight to amplify the gradient signal from the harder task. A weight of 10 was found to be effective.
```
total_loss = loss_cls + loss_reg * regression_weight
```
### Siamese Architecture
This project uses a Siamese (bi-encoder) architecture, which creates separate sentence embeddings for the premise and hypothesis using mean pooling. These vectors are then combined for the final prediction.

This approach was chosen over a simple cross-encoder (using [CLS]) for two reasons:

Performance: Literature suggests this method produces richer features ([vec_a, vec_b, |vec_a - vec_b|]) for complex regression and comparison tasks.

Fair Comparison: This architecture allows for a more direct and fair comparison between encoder models (BERT, RoBERTa) and decoder-only models (GPT-2), which cannot be used as a cross-encoder.

## 5. Results
Detailed experimental results, model comparisons (BERT vs. RoBERTa vs. GPT-2), and analysis are available in [**`Report.pdf`**](./Report.pdf).