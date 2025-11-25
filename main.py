from FileParser import FileParser
from constants import TEXT_COLUMN_NAME, TRAINING_DATA_PATH, MODEL_OPTIONS, VALUE_COLUMN_NAME, FINAL_DATA_PATH
from LanguageModel import LanguageModel
# from VaderBaseline import VaderBaseline 
from datasets import Dataset

if __name__ == '__main__':
  fileParser = FileParser()
  trainingData = fileParser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
  reducedTrainingData = {TEXT_COLUMN_NAME: trainingData[TEXT_COLUMN_NAME][0:100], VALUE_COLUMN_NAME: trainingData[VALUE_COLUMN_NAME][0:100]}
  # print(reducedTrainingData)
  # trainingDataSet = Dataset.from_dict(trainingData)
  trainingDataSet = Dataset.from_dict(reducedTrainingData)
  # print(trainingDataSet)
  #----------------------------
  # BERT / RoBERTa experiment
  # ----------------------------
  # languageModel = LanguageModel(MODEL_OPTIONS.SENTIMENT_ANALYSIS_BERT.value)
  # languageModel.train(trainingDataSet, trainingDataSet)
  # print(languageModel.test("This movie was bad"))
  # print(languageModel.test("This movie was good"))
  # print(languageModel.test("This movie was good, but I wouldn't care to see it again"))

  #----------------------------
  # LLaMA experiment
  # ----------------------------
  llama_model = LanguageModel(MODEL_OPTIONS.LLAMA_SENTIMENT.value)
  llama_model.train(trainingDataSet, trainingDataSet)
  print("LLaMA predictions:")
  print(llama_model.tes("This momive was bad"))
  print(llama_model.tes("This momive was good"))
  print(llama_model.tes("This momive was good, but I wouldn't care to see it again"))

  #----------------------------
  # VADER baseline (no training)
  # ----------------------------
  # vader = VaderBaseline()

  # sample_texts = [
  #   "This momive was bad",
  #   "This momive was good",
  #   "This momive was good, but I wouldn't care to see it again"
  # ]

  # print("\nBERT predictions:")
  # for t in sample_texts:
  #   print(t, "->", languageModel.test(t))

  # print("\nLLaMA predictions:")
  # for t in sample_texts:
  #   print(t, "->", llama_model.test(t))
  
  # print("\nVADER predictions (labels 0=neg,1=neu,2=pos):")
  # for t in sample_texts:
  #     print(t, "->", vader.predict_label(t))

  #----------------------------
  # evaluate VADER
  # ----------------------------
  # testData = fileParser.parseFile(FINAL_DATA_PATH + "test.csv", constantAdded=1)

  # vader = VaderBaseline()
  # preds = vader.predict_batch(testData[TEXT_COLUMN_NAME])
  # gold = testData[VALUE_COLUMN_NAME]

  # correct = sum(int(p==g) for p, g in zip(preds,gold))
  # accuracy = correct / len(gold)
  # print("VADER accuracy on test set:", accuracy)