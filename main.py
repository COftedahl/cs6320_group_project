from FileParser import FileParser
from constants import TEXT_COLUMN_NAME, TRAINING_DATA_PATH, MODEL_OPTIONS, VALUE_COLUMN_NAME, exampleSentences
from LanguageModel import LanguageModel
from datasets import Dataset

def test(languageModel):
  print(languageModel.test("This movie was bad"))
  print(languageModel.test("This movie was good"))
  print(languageModel.test("This movie was good, but I wouldn't care to see it again"))

def trainAndTest(string):
  fileParser = FileParser()
  trainingData = fileParser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
  reducedTrainingData = {TEXT_COLUMN_NAME: trainingData[TEXT_COLUMN_NAME][0:100], VALUE_COLUMN_NAME: trainingData[VALUE_COLUMN_NAME][0:100]}
  # print(reducedTrainingData)
  # trainingDataSet = Dataset.from_dict(trainingData)
  trainingDataSet = Dataset.from_dict(reducedTrainingData)
  # print(trainingDataSet)
  languageModel = LanguageModel(string)
  languageModel.train(trainingDataSet, trainingDataSet)
  test(languageModel)

if __name__ == '__main__':
  trainAndTest(MODEL_OPTIONS.SENTIMENT_ANALYSIS_BERT.value)
  # print(languageModel.test(exampleSentences[0]))
  # print(languageModel.test(exampleSentences[1]))
  # print(languageModel.test(exampleSentences[2]))
  # languageModel = LanguageModel(MODEL_OPTIONS.SENTIMENT_ANALYSIS_BERT.value)
  # languageModel.train(trainingDataSet, trainingDataSet)
  # print(languageModel.test("This movie was bad"))
  # print(languageModel.test("This movie was good"))
  # print(languageModel.test("This movie was good, but I wouldn't care to see it again"))