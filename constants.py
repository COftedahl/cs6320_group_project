from enum import Enum

TRAINING_DATA_PATH = "Data/"
VAL_DATA_PATH = "Data/"
class MODEL_OPTIONS(Enum):
  BERT_MULTILINGUAL = "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment"
  SENTIMENT_ANALYSIS_BERT = "https://huggingface.co/MarieAngeA13/Sentiment-Analysis-BERT"
  TWITTER_SENTIMENT_ANALYSIS_BERT = "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"