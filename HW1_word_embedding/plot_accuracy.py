import os
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt


# PREPROCESS = "remove_non_english_and_stop_words"
PREPROCESS = "remove_stop_words"

# --- Configuration ---
CHECKPOINTS_DIR = f"/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/checkpoints/{PREPROCESS}"
ANALOGY_FILE_PATH = "questions-words.csv"

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    """Calculates the accuracy, handling the case of empty arrays."""
    if gold.size == 0:
        return 0.0
    return np.mean(gold == pred)

def evaluate_model(model, data):
    preds = []
    golds = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating analogies"):
        analogy = row["Question"]
        words = analogy.lower().split()
        word1, word2, word3, word4 = words

        if not (model.wv.has_index_for(word1) and model.wv.has_index_for(word2) and model.wv.has_index_for(word3)):
            # print("skipping line word not exists in the corpus")
            # add wrong answer so the mask would work
            preds.append("a")
            golds.append("b")
            continue
        # Correct vector arithmetic: positive=[b, c], negative=[a]
        # The standard analogy task is d' = b - a + c
        result = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=1)
        pred = result[0][0]
        if (pred == word4):
            print(f"same: {pred}, {word4}")
        
        preds.append(pred)
        golds.append(word4)

    golds_np, preds_np = np.array(golds), np.array(preds)
    
    category_results = {}
    for category in data["Category"].unique():
        mask = data["Category"] == category
        golds_cat, preds_cat = golds_np[mask], preds_np[mask]
        acc_cat = calculate_accuracy(golds_cat, preds_cat)
        category_results[category] = acc_cat * 100
        print(f"Category: {category}, Accuracy: {acc_cat * 100}%")
        
    for sub_category in data["SubCategory"].unique():
        mask = data["SubCategory"] == sub_category
        golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
        acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
        category_results[sub_category] = acc_subcat * 100
        print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")
    
    return category_results

def plot_results(results):
    """Generates a bar chart from the evaluation results."""
    model_names = list(results.keys())
    semantic_scores = [res['Semantic'] for res in results.values()]
    syntactic_scores = [res['Syntactic'] for res in results.values()]

    x = np.arange(len(model_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 = ax.bar(x - width/2, semantic_scores, width, label='Semantic')
    rects2 = ax.bar(x + width/2, syntactic_scores, width, label='Syntactic')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Model Performance by Category (Preprocess: {PREPROCESS})')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    # Save the plot to a file
    plt.savefig(f"model_{PREPROCESS}.png")
    print("\nPlot saved as model_comparison.png")
    plt.show()


if __name__ == "__main__":
    # 1. Find all model files in the checkpoints directory
    checkpoints_path = Path(CHECKPOINTS_DIR)
    try:
        model_files = sorted([f for f in checkpoints_path.iterdir() if f.suffix == '.model'])
        if not model_files:
            print(f"Error: No '.model' files found in '{checkpoints_path}'")
            exit()
    except FileNotFoundError:
        print(f"Error: Checkpoints directory not found at '{checkpoints_path}'")
        exit()

    # 2. Load the analogy dataset
    try:
        analogy_data = pd.read_csv(ANALOGY_FILE_PATH)
        analogy_data.dropna(inplace=True) # Drop rows with missing values
    except FileNotFoundError:
        print(f"Error: Analogy file not found at '{ANALOGY_FILE_PATH}'")
        exit()

    # 3. Iterate through models, evaluate, and store results
    # model_files = ["/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/checkpoints/remove_stop_words/wiki_word2vec_20%_only_removed_stop_window=5_vs=100_sg=0_lr=0.025.model"]
    all_results = {}
    
    
    for model_path in model_files:
        # print(f"\n--- Evaluating model: {model_path.name} ---")
        
        model = Word2Vec.load(str(model_path))
        
        accuracy_dict = evaluate_model(model, analogy_data)
        results_filename = f"{model_path.stem}_results.txt"
        results_filepath = results_filename

        # 2. Write the contents of accuracy_dict to the text file
        try:
            with open(results_filepath, 'w') as f:
                f.write(f"Evaluation Results for Model: {model_path.name}\n")
                f.write("=" * 40 + "\n")
                for category, accuracy in accuracy_dict.items():
                    f.write(f"{category}: {accuracy:.4f}%\n")
            print(f"Successfully saved results to {results_filepath}")
        except IOError as e:
            print(f"Error: Could not write results to file {results_filepath}. Reason: {e}")
        
        # Store results with a shorter name for plotting
        # model_name = model_path.stem.replace('wiki_word2vec_', '')
        model_name = " "
        all_results[model_name] = accuracy_dict
        
    # 4. Plot the final results
    if all_results:
        plot_results(all_results)
    else:
        print("No results to plot.")
    
    
# Brown window=5_vs=100_sg=0_lr=0.025: Semantic 0.293, Syntatic 0.628