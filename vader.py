from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from constants import exampleSentences, SENTIMENT_RESULT_ENUM, SENTIMENT_RESULT_ARR, TRAINING_DATA_PATH, TEXT_COLUMN_NAME, VALUE_COLUMN_NAME
from FileParser import FileParser
from tqdm import tqdm

# https://www.geeksforgeeks.org/python/python-sentiment-analysis-using-vader/

THRESHOLD_VALUE = 0.05
POSITIVE_THRESHOLD = THRESHOLD_VALUE
NEGATIVE_THRESHOLD = -THRESHOLD_VALUE

def getOverallSentiment(sentiment): 
  if sentiment['compound'] >= POSITIVE_THRESHOLD:
    return SENTIMENT_RESULT_ENUM.POSITIVE
  elif sentiment['compound'] <= NEGATIVE_THRESHOLD:
    return SENTIMENT_RESULT_ENUM.NEGATIVE
  else:
    return SENTIMENT_RESULT_ENUM.NEUTRAL
  
def computeMetrics(predictions, trainingData): 
  numTests = len(predictions)
  numCorrectPredictions = 0
  numCorrectPositives = 0
  numFalsePositives = 0
  numFalseNegatives = 0
  for i in tqdm(range(0, numTests)): 
    # print("Prediction:" + str(predictions[i].value) + " -- Truth Value: " + str(trainingData[VALUE_COLUMN_NAME][i]))
    if (predictions[i].value == trainingData[VALUE_COLUMN_NAME][i]): 
      # the prediction was correct
      numCorrectPredictions += 1
      if (predictions[i].value == SENTIMENT_RESULT_ENUM.POSITIVE.value): 
        numCorrectPositives += 1
    else: 
      # the prediction was wrong
      if (predictions[i].value == SENTIMENT_RESULT_ENUM.NEGATIVE.value): 
        numFalseNegatives += 1
      elif (predictions[i].value == SENTIMENT_RESULT_ENUM.POSITIVE.value): 
        numFalsePositives += 1
  recall = numCorrectPositives / (numCorrectPositives + numFalseNegatives)
  accuracy = numCorrectPredictions / numTests
  precision = numCorrectPositives / (numCorrectPositives + numFalsePositives)
  f1 = (2 * precision * recall) / (precision + recall)
  return {"number of predictions: ": numTests, "recall": recall, "accuracy": accuracy, "precision": precision, "f1": f1}

if __name__ == '__main__':
  fileParser = FileParser()
  trainingData = fileParser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
  # sentences = trainingData
  analyzer = SentimentIntensityAnalyzer()
  predictions = []
  print("Predicting labels...")
  
  for sentence in tqdm(trainingData[TEXT_COLUMN_NAME]):
    vs = analyzer.polarity_scores(sentence)
    label = getOverallSentiment(vs)
    # print("{:-<65} Result: {}, Compund Score: {}".format(sentence, str(SENTIMENT_RESULT_ARR[label.value]), str(vs['compound'])))
    predictions.append(label)
  print("Generating metrics...")
  print(computeMetrics(predictions, trainingData))