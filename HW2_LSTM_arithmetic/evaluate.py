import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
from tqdm import tqdm

# --- Import from shared model file ---
from model import CharRNN, EMBED_DIM, HIDDEN_DIM

def generator(model, start_char, max_len, char_to_id, id_to_char):
    """
    Generates a sequence from a model given a starting prompt.
    This function now takes the model as an argument.
    """
    generated_string = start_char
    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            char_ids = [char_to_id[c] for c in generated_string]
            input_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0)
            input_tensor = input_tensor.to(next(model.parameters()).device)

            # Since the generator is not part of the model class, we access modules directly
            embedded = model.embedding(input_tensor)
            rnn_out, _ = model.rnn_layer1(embedded)
            rnn_out, _ = model.rnn_layer2(rnn_out)
            last_token_logits = model.linear(rnn_out[:, -1, :])

            next_char_id = torch.argmax(last_token_logits, dim=1).item()
            next_char = id_to_char[next_char_id]

            generated_string += next_char

            if next_char == '<eos>':
                break
    return generated_string

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description='Evaluate an LSTM for arithmetic tasks.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pth).')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the dataset files.')
    args = parser.parse_args()

    # --- 2. Setup Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 3. Load Data and Build Dictionary ---
    data_path = Path(args.data_dir)
    try:
        df_train = pd.read_csv(data_path / 'arithmetic_train.csv')
        df_eval = pd.read_csv(data_path / 'arithmetic_eval.csv')
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. {e}")
        print(f"Please ensure 'arithmetic_train.csv' and 'arithmetic_eval.csv' are in '{data_path}'")
        return

    # Rebuild dictionary from training data to ensure consistency
    char_to_id = {'<pad>': 0, '<eos>': 1}
    char_id_counter = 2
    df_train['tgt'] = df_train['tgt'].apply(str)
    for seq in (df_train['src'] + df_train['tgt']):
        for char in seq:
            if char not in char_to_id:
                char_to_id[char] = char_id_counter
                char_id_counter += 1
    id_to_char = {v: k for k, v in char_to_id.items()}
    vocab_size = len(char_to_id)
    padding_idx = char_to_id['<pad>']
    print(f"Vocabulary size: {vocab_size}")

    # Prepare evaluation data
    df_eval['tgt'] = df_eval['tgt'].apply(str)
    df_eval['full_answer'] = df_eval['src'] + df_eval['tgt'] + '<eos>'
    df_eval['prompt'] = df_eval['src']
    max_gen_len = df_eval['full_answer'].str.len().max() + 5 # Set a safe max generation length

    # --- 4. Load Model ---
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = CharRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, padding_idx).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Please ensure the model architecture and hyperparameters in this script match the saved checkpoint.")
        return
    
    model.eval()

    # --- 5. Run Evaluation ---
    matched = 0
    incorrect_predictions = []
    correct_predictions = []

    print(f"\nRunning evaluation on {len(df_eval)} samples...")
    with torch.no_grad():
        for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Evaluating"):
            prompt = row['prompt']
            ground_truth = row['full_answer']
            
            prediction = generator(model, prompt, max_gen_len, char_to_id, id_to_char)
            
            if prediction == ground_truth:
                matched += 1
                if len(correct_predictions) < 5:
                    correct_predictions.append(f"  Prompt: {prompt}\n  Prediction: {prediction}\n")
            else:
                incorrect_predictions.append(f"  Prompt: {prompt}\n  Truth: {ground_truth}\n  Prediction: {prediction}\n")

    # --- 6. Report Results ---
    em_accuracy = matched / len(df_eval)
    print("\n--- Evaluation Complete ---")
    print(f"Exact Match Accuracy: {em_accuracy:.4f} ({matched}/{len(df_eval)})")

    print("\n--- Sample Correct Predictions ---")
    if correct_predictions:
        print("\n".join(correct_predictions))
    else:
        print("No correct predictions in the sample.")

    print("\n--- Sample Incorrect Predictions ---")
    if incorrect_predictions:
        print("\n".join(incorrect_predictions))
    else:
        print("No incorrect predictions in the sample.")

if __name__ == '__main__':
    main()