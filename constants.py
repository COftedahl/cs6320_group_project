from enum import Enum

RAW_DATA_PATH = "./Data/clean/"
CLEANED_DATA_PATH = "./Data/clean/"
FINAL_DATA_PATH = "./Data/final/"
# TRAINING_DATA_PATH = CLEANED_DATA_PATH + "Sentiment_Analysis.csv"
TRAINING_DATA_PATH = FINAL_DATA_PATH + "train.csv"
VAL_DATA_PATH = ""
USER_AT_TOKEN = "<atToken>"
TEXT_COLUMN_NAME = "text"
VALUE_COLUMN_NAME = "label"

class MODEL_OPTIONS(Enum):
  BERT_MULTILINGUAL = "nlptown/bert-base-multilingual-uncased-sentiment" # "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment"
  SENTIMENT_ANALYSIS_BERT = "MarieAngeA13/Sentiment-Analysis-BERT" # "https://huggingface.co/MarieAngeA13/Sentiment-Analysis-BERT"
  TWITTER_SENTIMENT_ANALYSIS_BERT = "cardiffnlp/twitter-roberta-base-sentiment-latest" # "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"
  LLAMA_SENTIMENT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Supports 3-class sentiment, and is small/light, perfect for training and inference




# https://solutionfall.com/question/why-is-the-model-not-returning-loss-from-inputs-in-trainertrain-resulting-in-a-valueerror/#:~:text=To%20address%20this%20issue%20and%20ensure%20that%20the,returns%20the%20loss%20value%20in%20the%20%60compute_loss%60%20function.