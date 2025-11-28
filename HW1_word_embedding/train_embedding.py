import logging
from gensim.models.callbacks import CallbackAny2Vec

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define a custom callback to print loss after each epoch
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_per_epoch = [0]

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch + 1} begins.")

    def on_epoch_end(self, model):
        # The loss can be accessed via model.get_latest_training_loss()
        # Note: This method might not be available in all gensim versions
        try:
            current_loss = model.get_latest_training_loss()
            epoch_loss = current_loss - self.loss_per_epoch[-1]
            self.loss_per_epoch.append(current_loss)
            print(f"Epoch #{self.epoch + 1} ends. Loss: {epoch_loss:.4f}")
        except AttributeError:
            print(f"Epoch #{self.epoch + 1} ends. (Loss not available)")
        self.epoch += 1

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

vector_size = 100  # Dimensionality of the word vectors default: 100
window = 5         # context window default: 5
min_count = 5      
workers = multiprocessing.cpu_count() 
sg = 0             # Training algorithm: 1 for skip-gram; otherwise CBOW; default: 1
alpha = 0.025      # lr, default: 0.025
sample_percent = 20 # for saved checkpoint consistency 

# corpus_path = "/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/data/preprocessed_wiki_corpus/wiki_sample_10%_en_only_removed_stop.txt"
corpus_path = f"/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/data/preprocessed_wiki_corpus/wiki_sample_{sample_percent}%_removed_stop.txt"
# corpus_path = "/home/cvlab123/Kyle_Having_Fun/NLP_nthu/HW1_word_embedding/data/brown_corpus/brown_corpus_removed_stop.txt"
checkpoint_path = f"checkpoints/remove_stop_words/wiki_word2vec_{sample_percent}%_only_removed_stop_window={window}_vs={vector_size}_sg={sg}_lr={alpha}.model"
# checkpoint_path = f"checkpoints/remove_stop_words/wiki_word2vec_Brown_window={window}_vs={vector_size}_sg={sg}_lr={alpha}.model"

sentences = SentenceIterator(corpus_path)

epoch_logger = EpochLogger()

# Train the model with the callback
print("Training Word2Vec model... This may take some time.")
model = Word2Vec(
    sentences=sentences,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    sg=sg,
    epochs=10, # You should specify the number of epochs
    compute_loss=True, # Important: enable loss computation
    alpha=alpha,
    callbacks=[epoch_logger]
)
print("✅ Training complete.")

# --- 4. Save the model ---
model.save(checkpoint_path)
print(f"Model saved to: {checkpoint_path}")


# nohup python train_embedding.py > train_5%.log 2>&1 &