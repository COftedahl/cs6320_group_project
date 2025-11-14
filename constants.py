from enum import Enum

RAW_DATA_PATH = "./data/raw/"
CLEANED_DATA_PATH = "./data/clean/"
FINAL_DATA_PATH = "./data/final/"
USER_AT_TOKEN = "<atToken>"
class MODEL_OPTIONS(Enum):
  BERT_MULTILINGUAL = "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment"
  SENTIMENT_ANALYSIS_BERT = "https://huggingface.co/MarieAngeA13/Sentiment-Analysis-BERT"
  TWITTER_SENTIMENT_ANALYSIS_BERT = "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"