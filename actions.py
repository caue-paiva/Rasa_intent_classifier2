from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

import spacy
print(spacy.load('pt_core_news_lg').path)


print("Starting script...")

print("Loading data...")
train_data = load_data('/home/kap/Desktop/RASA_NLP3/rasa_data_pt2.json')

print("Creating trainer...")
trainer = Trainer(config.load("config_spacy.yaml"))

print("Training model...")
trainer.train(train_data)

print("Persisting model...")
model_directory = trainer.persist("models")

print("Model saved at: ", model_directory)


