from FileParser import FileParser
from constants import TRAINING_DATA_PATH, VAL_DATA_PATH, MODEL_OPTIONS
from LanguageModel import LanguageModel
from datasets import Dataset

if __name__ == '__main__':
  fileParser = FileParser()
  trainingData = fileParser.parseFile(TRAINING_DATA_PATH)
  # reducedTrainingData = {"text": trainingData["text"][0:100], "values": trainingData["values"][0:100]}
  # print(reducedTrainingData)
  trainingDataSet = Dataset.from_dict(trainingData)
  print(trainingDataSet)
  lm = LanguageModel(MODEL_OPTIONS.BERT_MULTILINGUAL.value)
  lm.train(trainingDataSet, trainingDataSet)