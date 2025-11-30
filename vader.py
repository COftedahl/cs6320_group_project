from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from constants import exampleSentences, SENTIMENT_RESULT_ENUM, SENTIMENT_RESULT_ARR, TRAINING_DATA_PATH, TEXT_COLUMN_NAME, VALUE_COLUMN_NAME, VAL_DATA_PATH, TEST_DATA_PATH, RESULTS_DIR, SAVED_MODEL_RESULTS
from FileParser import FileParser
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
# from sklearn.metrics import confusion_matrix

# https://www.geeksforgeeks.org/python/python-sentiment-analysis-using-vader/

def make_confusion_matrix(model_name, cm):
  labels = ['Negative', 'Neutral', 'Positive']
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=labels, yticklabels=labels, robust=True)
  plt.title(f'Confusion Matrix - {model_name}')
  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  plt.tight_layout()
  plt.savefig(RESULTS_DIR + model_name + "_confusion_matrix.png")
  plt.close()

  print(f"Saved results to {RESULTS_DIR}")

def make_comparison_plot(all_results):
  """Makes bar chart comparing model accuracies"""
  names = [r["model_name"] for r in all_results]
  accs = [r["test_accuracy"] for r in all_results]

  plt.figure(figsize=(10, 6))
  bars = plt.bar(names, accs)
  plt.ylabel('Test Accuracy')
  plt.title('Model Comparison')
  plt.ylim(0, 1)

  # Add labels on bars
  for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{acc:.3f}', ha='center')

  plt.tight_layout()
  plt.savefig(RESULTS_DIR + "model_comparison.png")
  plt.close()
  print("Saved comparison plot")

THRESHOLD_VALUE = 0.5
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
  # numCorrectPredictions = 0
  # numCorrectPositives = 0
  # numFalsePositives = 0
  # numFalseNegatives = 0

  predictionsForTruePosLabel = {SENTIMENT_RESULT_ENUM.NEGATIVE.value: 0, SENTIMENT_RESULT_ENUM.NEUTRAL.value: 0, SENTIMENT_RESULT_ENUM.POSITIVE.value: 0}
  predictionsForTrueNeuLabel = {SENTIMENT_RESULT_ENUM.NEGATIVE.value: 0, SENTIMENT_RESULT_ENUM.NEUTRAL.value: 0, SENTIMENT_RESULT_ENUM.POSITIVE.value: 0}
  predictionsForTrueNegLabel = {SENTIMENT_RESULT_ENUM.NEGATIVE.value: 0, SENTIMENT_RESULT_ENUM.NEUTRAL.value: 0, SENTIMENT_RESULT_ENUM.POSITIVE.value: 0}

  for i in tqdm(range(0, numTests)): 
    # print("Prediction:" + str(predictions[i].value) + " -- Truth Value: " + str(trainingData[VALUE_COLUMN_NAME][i]))
    # if (predictions[i].value == trainingData[VALUE_COLUMN_NAME][i]): 
    #   # the prediction was correct
    #   numCorrectPredictions += 1
    #   if (predictions[i].value == SENTIMENT_RESULT_ENUM.POSITIVE.value): 
    #     numCorrectPositives += 1
    # else: 
    #   # the prediction was wrong
    #   if (predictions[i].value == SENTIMENT_RESULT_ENUM.NEGATIVE.value): 
    #     numFalseNegatives += 1
    #   elif (predictions[i].value == SENTIMENT_RESULT_ENUM.POSITIVE.value): 
    #     numFalsePositives += 1

    if (trainingData[VALUE_COLUMN_NAME][i] == SENTIMENT_RESULT_ENUM.POSITIVE.value): 
      predictionsForTruePosLabel[predictions[i].value] += 1
    elif (trainingData[VALUE_COLUMN_NAME][i] == SENTIMENT_RESULT_ENUM.NEUTRAL.value): 
      predictionsForTrueNeuLabel[predictions[i].value] += 1
    else: 
      predictionsForTrueNegLabel[predictions[i].value] += 1
  numCorrectPositives = predictionsForTruePosLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value]
  numFalseNegatives = predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value] + predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.NEUTRAL.value]
  numCorrectPredictions = predictionsForTruePosLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value] + predictionsForTrueNeuLabel[SENTIMENT_RESULT_ENUM.NEUTRAL.value] + predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.NEGATIVE.value]
  numFalsePositives = predictionsForTrueNeuLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value] + predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value]

  # cm = confusion_matrix(trainingData[VALUE_COLUMN_NAME], predictions)
  cm = [
    [
      predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.NEGATIVE.value], predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.NEUTRAL.value], predictionsForTrueNegLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value], 
    ], 
    [
      predictionsForTrueNeuLabel[SENTIMENT_RESULT_ENUM.NEGATIVE.value], predictionsForTrueNeuLabel[SENTIMENT_RESULT_ENUM.NEUTRAL.value], predictionsForTrueNeuLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value], 
    ], 
    [
      predictionsForTruePosLabel[SENTIMENT_RESULT_ENUM.NEGATIVE.value], predictionsForTruePosLabel[SENTIMENT_RESULT_ENUM.NEUTRAL.value], predictionsForTruePosLabel[SENTIMENT_RESULT_ENUM.POSITIVE.value], 
    ]
  ]
  print(cm)
  scaleFactor = 100
  cmRowSums = [sum(row) for row in cm]

  rowIndex = 0
  rowNormalizedCM = []
  for row in cm: 
    rowNormalizedCM.append([int(scaleFactor * entry / cmRowSums[rowIndex]) for entry in row])
    rowIndex += 1

  print(rowNormalizedCM)
  make_confusion_matrix("VADER", rowNormalizedCM)

  recall = numCorrectPositives / (numCorrectPositives + numFalseNegatives)
  accuracy = numCorrectPredictions / numTests
  precision = numCorrectPositives / (numCorrectPositives + numFalsePositives)
  f1 = (2 * precision * recall) / (precision + recall)
  return {"number of predictions: ": numTests, "recall": recall, "accuracy": accuracy, "precision": precision, "f1": f1, "confusion matrix": {
    "Predictions for true negative label": predictionsForTrueNegLabel, 
    "Predictions for true netural label": predictionsForTrueNeuLabel, 
    "Predictions for true positive label": predictionsForTruePosLabel, 
  }}

def saveToFile(path, content): 
  try: 
    with open (path, 'x') as file: 
      file.write(content)
  except FileExistsError: 
    with open (path, 'w') as file: 
      file.write(content)
  except Exception as e:
    print(e)

if __name__ == '__main__':
  fileParser = FileParser()
  trainingData1 = fileParser.parseFile(TRAINING_DATA_PATH, constantAdded=1)
  trainingData2 = fileParser.parseFile(TEST_DATA_PATH, constantAdded=1)
  trainingData3 = fileParser.parseFile(VAL_DATA_PATH, constantAdded=1)
  trainingData = dict({TEXT_COLUMN_NAME: trainingData1[TEXT_COLUMN_NAME] + trainingData2[TEXT_COLUMN_NAME] + trainingData3[TEXT_COLUMN_NAME], 
                       VALUE_COLUMN_NAME: trainingData1[VALUE_COLUMN_NAME] + trainingData2[VALUE_COLUMN_NAME] + trainingData3[VALUE_COLUMN_NAME]})
  # sentences = trainingData
  print("first 10 training strings: " + str(trainingData[TEXT_COLUMN_NAME][:10]))
  analyzer = SentimentIntensityAnalyzer()
  predictions = []
  print("Predicting labels...")
  
  for sentence in tqdm(trainingData[TEXT_COLUMN_NAME]):
    # print("testing sentence: " + str(sentence))
    vs = analyzer.polarity_scores(sentence)
    label = getOverallSentiment(vs)
    # print("{:-<65} Result: {}, Compund Score: {}".format(sentence, str(SENTIMENT_RESULT_ARR[label.value]), str(vs['compound'])))
    predictions.append(label)
  print("Generating metrics...")
  metrics = computeMetrics(predictions, trainingData)
  print(metrics)
  saveToFile("./results/vader.txt", str(metrics))