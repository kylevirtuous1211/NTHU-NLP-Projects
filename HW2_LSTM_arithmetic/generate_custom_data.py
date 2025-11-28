import os
import numpy as np
import pandas as pd


def generate_arithmetic_data(num_samples, min_val, max_val):
    """
    Generates a DataFrame with arithmetic problems (a+b=c).
    
    Args:
        num_samples (int): The number of samples to generate.
        min_val (int): The minimum value for the operands.
        max_val (int): The maximum value for the operands.

    Returns:
        pd.DataFrame: A DataFrame with 'src' and 'tgt' columns.
    """
    data = []
    for _ in range(num_samples):
        a = np.random.randint(min_val, max_val + 1)
        b = np.random.randint(min_val, max_val + 1)
        
        # Correctly sample from all 4 operators
        op_id = np.random.randint(low=0, high=4)
        op = ['+', '-', '*', '/'][op_id]

        # Handle division by zero and use integer division
        if op == '/' and b == 0:
            # Avoid division by zero by re-sampling or skipping
            continue # Simple solution: skip this sample

        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            result = a // b # Use integer division
        
        src = f"{a}{op}{b}="
        tgt = str(result)
        data.append({'src': src, 'tgt': tgt})
        
    return pd.DataFrame(data)

def create_3digit_train_2digit_eval(output_dir='./data_custom'):
    """
    Scenario 1: Creates a training set with 3-digit numbers and an
    evaluation set with 2-digit numbers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating 3-digit training data (e.g., 100+100 to 999+999)...")
    df_train_3_digit = generate_arithmetic_data(num_samples=1048576, min_val=100, max_val=999)
    train_path = os.path.join(output_dir, 'arithmetic_train_3digit.csv')
    df_train_3_digit.to_csv(train_path, index=False)
    print(f"Saved 3-digit training data to {train_path}")

    print("\nGenerating 2-digit evaluation data (e.g., 10+10 to 99+99)...")
    df_eval_2_digit = generate_arithmetic_data(num_samples=263251, min_val=10, max_val=99)
    eval_path = os.path.join(output_dir, 'arithmetic_eval_2digit.csv')
    df_eval_2_digit.to_csv(eval_path, index=False)
    print(f"Saved 2-digit evaluation data to {eval_path}")

def create_noisy_training_data(input_file, output_file, noise_ratio=0.2):
    """
    Loads an existing dataset, introduces noise into the 'tgt' column,
    and saves it to a new file.

    Args:
        input_file (str): Path to the source .csv file (e.g., 'data/arithmetic_train.csv').
        output_file (str): Path to save the new noisy .csv file.
        noise_ratio (float): The ratio of answers to make incorrect.
    """
    try:
        print(f"Loading data from {input_file}...")
        df_noisy = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    print(f"Introducing {noise_ratio*100}% noise into the 'tgt' column...")
    # Get indices to corrupt
    num_to_corrupt = int(len(df_noisy) * noise_ratio)
    indices_to_corrupt = np.random.choice(df_noisy.index, num_to_corrupt, replace=False)
    
    # Introduce noise
    for idx in indices_to_corrupt:
        correct_answer = str(df_noisy.loc[idx, 'tgt'])
        # Generate a random incorrect answer of the same length
        len_answer = len(correct_answer)
        min_rand = 10**(len_answer - 1) if len_answer > 1 else 0
        max_rand = (10**len_answer) - 1 if len_answer > 0 else 9
        incorrect_answer = str(np.random.randint(min_rand, max_rand + 1))
        
        # Ensure the random answer is not the correct one
        while incorrect_answer == correct_answer:
            incorrect_answer = str(np.random.randint(min_rand, max_rand + 1))
        
        df_noisy.loc[idx, 'tgt'] = incorrect_answer
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_noisy.to_csv(output_file, index=False)
    print(f"Saved noisy training data to {output_file}")

if __name__ == '__main__':
    # create_3digit_train_2digit_eval()
    create_noisy_training_data(
        input_file='data/arithmetic_train.csv',
        output_file='data/arithmetic_train_20%noisy.csv',
        noise_ratio=0.2
    )