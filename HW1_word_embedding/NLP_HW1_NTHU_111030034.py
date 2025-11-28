# %% [markdown]
# ## Part I: Data Pre-processing

# %%
import pandas as pd

# %%
# Download the Google Analogy dataset
# !wget http://download.tensorflow.org/data/questions-words.txt

# %%
# Preprocess the dataset
file_name = "questions-words"
with open(f"{file_name}.txt", "r") as f:
    data = f.read().splitlines()

# %%
# check data from the first 10 entries
for entry in data[:10]:
    print(entry)

# %%
# TODO1: Write your code here for processing data to pd.DataFrame
# Please note that the first five mentions of ": " indicate `semantic`,
# and the remaining nine belong to the `syntatic` category.
questions = []
categories = []
sub_categories = []

counter = 0
current_category = ""
current_sub = ""
for entry in data:
    if entry[0] == ":":
        counter += 1
        current_sub = entry
        if counter < 6:
            current_category = "Semantic"
        else:
            current_category = "Syntactic"
        continue
    questions.append(entry)
    categories.append(current_category)
    sub_categories.append(current_sub)


# %%
# Create the dataframe
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

# %%
df.head(12146)

# %%
df.to_csv(f"{file_name}.csv", index=False)

# %% [markdown]
# ## Part II: Use pre-trained word embeddings
# - After finish Part I, you can run Part II code blocks only.

# %%
import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import multiprocessing
from gensim.models import Word2Vec

# %%
data = pd.read_csv("questions-words.csv")

# %%
MODEL_NAME = "glove-wiki-gigaword-100"
# You can try other models.
# https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models

# Load the pre-trained model (using GloVe vectors here)
model_pretrained = gensim.downloader.load(MODEL_NAME)
print("The Gensim model loaded successfully!")

# %%
# Do predictions and preserve the gold answers (word_D)
preds_pre = []
golds_pre = []

# for analogy in tqdm(data["Question"]):
#       # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
#       # You should also preserve the gold answers during iterations for evaluations later.
#       """ Hints
#       # Unpack the analogy (e.g., "man", "woman", "king", "queen")
#       # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
#       # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
#       # Mikolov et al., 2013: big - biggest and small - smallest
#       # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
#       """
#       words = analogy.lower().split()
#       word1, word2, word3, word4 = words
#       result = model_pretrained.most_similar(positive=[word2, word3], negative=[word1], topn=1)
#       pred = result[0][0]
#       if (pred == word4):
#             print(f"predicted: {pred}, GT: {word4}")
#       preds_pre.append(pred)
#       golds_pre.append(word4)
      

# %%
# Perform evaluations. You do not need to modify this block!!

# def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
#     return np.mean(gold == pred)

# golds_np, preds_np = np.array(golds_pre), np.array(preds_pre)
# data = pd.read_csv("questions-words.csv")

# # Evaluation: categories
# for category in data["Category"].unique():
#     mask = data["Category"] == category
#     golds_cat, preds_cat = golds_np[mask], preds_np[mask]
#     acc_cat = calculate_accuracy(golds_cat, preds_cat)
#     print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# # Evaluation: sub-categories
# for sub_category in data["SubCategory"].unique():
#     mask = data["SubCategory"] == sub_category
#     golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
#     acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
#     print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO3: Plot t-SNE for the words in the SUB_CATEGORY `: family`
family_words = set()
questions = data["Question"][data["SubCategory"] == SUB_CATEGORY]
for q in questions:
    for t in q.split():
        family_words.add(t)

family_word_np = np.array(list(family_words))

family_embedding = []
for w in family_word_np:
    family_embedding.append(model_pretrained[w])

family_embedding = np.array(family_embedding)

tsne = TSNE(
    n_components=2,
    random_state=69,
    perplexity=30,
)

embedding2d = tsne.fit_transform(family_embedding)
plt.scatter(embedding2d[:,0], embedding2d[:,1], c='steelblue')

for text, (x, y) in zip(family_words, embedding2d):
    plt.annotate(
        text=text,
        xy=(x, y),
        xytext=(3,3),
        textcoords='offset points',
        fontsize='x-small'
    )

plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")

# %% [markdown]
# ### Part III: Train your own word embeddings

# %% [markdown]
# ### Get the latest English Wikipedia articles and do sampling.
# - Usually, we start from Wikipedia dump (https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). However, the downloading step will take very long. Also, the cleaning step for the Wikipedia corpus ([`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus)) will take much time. Therefore, we provide cleaned files for you.

# %%
# # Download the split Wikipedia files
# # Each file contain 562365 lines (articles).
# !gdown --id 1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd -O wiki_texts_part_0.txt.gz
# !gdown --id 1ABblLRd9HXdXvaNv8H9fFq984bhnowoG -O wiki_texts_part_1.txt.gz
# !gdown --id 1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M -O wiki_texts_part_2.txt.gz
# !gdown --id 1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g -O wiki_texts_part_3.txt.gz
# !gdown --id 16mBeG26m9LzHXdPe8UrijUIc6sHxhknz -O wiki_texts_part_4.txt.gz

# %%
# # Download the split Wikipedia files
# # Each file contain 562365 lines (articles), except the last file.
# !gdown --id 17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI -O wiki_texts_part_5.txt.gz
# !gdown --id 19IvB2vOJRGlrYulnTXlZECR8zT5v550P -O wiki_texts_part_6.txt.gz
# !gdown --id 1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV -O wiki_texts_part_7.txt.gz
# !gdown --id 1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX -O wiki_texts_part_8.txt.gz
# !gdown --id 17eQXcrvY1cfpKelLbP2BhQKrljnFNykr -O wiki_texts_part_9.txt.gz
# !gdown --id 1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62 -O wiki_texts_part_10.txt.gz

# %%
# Extract the downloaded wiki_texts_parts files.
# !gunzip -k wiki_texts_part_*.gz

# %%
# Combine the extracted wiki_texts_parts files.
# !cat wiki_texts_part_*.txt > wiki_texts_combined.txt

# %%
# Check the first ten lines of the combined file
# !head -n 10 wiki_texts_combined.txt

# %% [markdown]
# Please note that we used the default parameters of [`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus) for cleaning the Wiki raw file. Thus, words with one character were discarded.

# %%
# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.
import os
import random

wiki_txt_path = "wiki_texts_combined.txt"
# wiki_texts_combined.txt is a text file separated by linebreaks (\n).
# Each row in wiki_texts_combined.txt indicates a Wikipedia article.
wiki_dir = "data/wikidata"
wiki_txt_path = os.path.join(wiki_dir, wiki_txt_path)
random.seed(42)

SAMPLE_RATE = 0.05
output_dir = "data/preprocessed_wiki_corpus"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"wiki_sample_{int(SAMPLE_RATE*100)}%.txt")
with open(wiki_txt_path, "r", encoding="utf-8") as f:
    with open(output_path, "a", encoding="utf-8") as output_file:
    # TODO4: Sample `20%` Wikipedia articles
        for line in f:
            if (random.random() < SAMPLE_RATE):
                output_file.write(line)

    

# %%
# # iniialize stuff for translation task
# import string 
# import nltk
# TRANSLATOR = str.maketrans('', '', string.punctuation)
# try:
#     ENGLISH_WORDS = set(nltk.corpus.words.words())
# except:
#     print("nltk words not found, creating now")
#     nltk.download('words')
#     ENGLISH_WORDS = set(nltk.corpus.words.words())

# %%
# import nltk
# from nltk.corpus import brown

# # Download the corpus (only needs to be done once)
# nltk.download('gutenberg')

# # You can access the sentences or words directly
# sentences = brown.sents()
# words = brown.words()

# # You can write it to a file to train your model
# with open('data/gutenberg/gutenberg_corpus.txt', 'w') as f:
#     for sentence in sentences:
#         f.write(' '.join(sentence) + '\n')

# %%
# import pytorch

# TODO5: Train your own word embeddings with the sampled articles
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# Hint: You should perform some pre-processing before training.

# preprocess .txt corpus
# 1. remove non-english words
# nltk.download('words') # download the corpus if not already
# def remove_non_english(text) -> str:
#     # create filter
#     text = text.translate(TRANSLATOR)
#     # remove punctuation like . , : 
#     text_english_list = [word for word in text.split() if word in ENGLISH_WORDS]
#     return  ' '.join(text_english_list)

def remove_stop_words(text) -> str:
    common_stop_words = set([
        "a", "an", "the", "in", "on", "at", "for", "to", "of", "with", "by",
        "is", "am", "are", "was", "were", "be", "and", "but", "or", "so",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "what", "which", "who", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "will", "just", "don", "should", "now"
    ])
    removed_stop_words = [word for word in text.split() if word not in common_stop_words]
    return ' '.join(removed_stop_words)

from pathlib import Path
# original_corpus = "/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/data/brown_corpus/brown_corpus.txt"
original_corpus = Path(output_path)
output_corpus_en = original_corpus.stem + "_removed_stop.txt"

print(f"translating {original_corpus},\n outputing to {output_corpus_en}")

with open(original_corpus, "r") as infile, open(output_corpus_en, "w") as outfile:
    for line in infile:
        # processed_line = remove_non_english(line)
        processed_line_removed_stop = remove_stop_words(line)
        if (processed_line_removed_stop):
            outfile.write(processed_line_removed_stop + '\n')
            
        

# %%
# line iterator
class SentenceIterator:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        with open (self.filepath, "r") as f:
            for line in f:
                yield line.split()
                

import multiprocessing
from gensim.models import Word2Vec

vector_size = 100  # Dimensionality of the word vectors
window = 10         # context window
min_count = 5      # Ignores all words with a total frequency lower than this
workers = multiprocessing.cpu_count() 
sg = 1             # Training algorithm: 1 for skip-gram; otherwise CBOW.

# corpus_path = "/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/data/preprocessed_wiki_corpus/wiki_sample_20%_en_only_removed_stop.txt"
# checkpoint_path = f"checkpoints/wiki_word2vec_window={window}_vs={vector_size}_sg={sg}.model"
corpus_path = output_corpus_en

sentences = SentenceIterator(corpus_path)

print("Training Word2Vec model... This may take some time.")
model_mine = Word2Vec(
    sentences=sentences,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    sg=sg
)
print("✅ Training complete.")

# --- 4. Save the model_mine ---
# model_mine.save(checkpoint_path)
# print(f"Model saved to: {checkpoint_path}")

# %%
# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
      # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      words = analogy.lower().split()
      word1, word2, word3, word4 = words
      if not (model_mine.wv.has_index_for(word1) and model_mine.wv.has_index_for(word2) and model_mine.wv.has_index_for(word3)):
            # print("skipping line word not exists in the corpus")
            # add wrong answer so the mask would work
            preds.append("a")
            golds.append("b")
            continue
      
      result = model_mine.wv.most_similar(positive=[word2, word3], negative=[word1], topn=1)
      # print(result)
      pred = result[0][0]
      # print(f"predicted: {pred}, GT: {word4}")
      if (pred == word4) :
            print(f"prediction correct: {pred}")
      
      preds.append(pred)
      golds.append(word4)

# %%
def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"
family_words = set()
questions = data["Question"][data["SubCategory"] == SUB_CATEGORY]
for q in questions:
    for t in q.split():
        family_words.add(t)

family_words_in_vocab = [word for word in family_words if word in model_mine.wv]

family_embedding = []
for w in family_words_in_vocab:
    family_embedding.append(model_mine.wv[w])

family_embedding = np.array(family_embedding)

tsne = TSNE(
    n_components=2,
    random_state=69,
    perplexity=12,
)

embedding2d = tsne.fit_transform(family_embedding)
plt.scatter(embedding2d[:,0], embedding2d[:,1], c='steelblue')

for text, (x, y) in zip(family_words, embedding2d):
    plt.annotate(
        text=text,
        xy=(x, y),
        xytext=(3,3),
        textcoords='offset points',
        fontsize='x-small'
    )
    
plt.title("Word Relationships from HW1 word analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")

# %%
# model_path = "/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/checkpoints/remove_stop_words/wiki_word2vec_20%_only_removed_stop_window=5_vs=100_sg=0_lr=0.025.model"
# model_mine = Word2Vec.load(str(model_path))

word = "water"
similar_words = model_mine.wv.most_similar(word, topn=5)
for similar_word, score in similar_words:
    print(f"    {similar_word:<15} (Score: {score:.4f})")

# %%


# %%



