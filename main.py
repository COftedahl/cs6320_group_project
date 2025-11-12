from FileParser import FileParser
from constants import TRAINING_DATA_PATH, VAL_DATA_PATH, MODEL_OPTIONS
from LanguageModel import LanguageModel

if __name__ == '__main__':
  fileParser = FileParser()
  trainingData = fileParser.parseFile(TRAINING_DATA_PATH)
  lm = LanguageModel(MODEL_OPTIONS.BERT_MULTILINGUAL)
  lm.train(trainingData)