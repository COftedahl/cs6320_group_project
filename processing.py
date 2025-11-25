import constants
import re
import math

def combinedPrintLine(text, sentementVal, file):
    outFile = open(constants.CLEANED_DATA_PATH + file, "a")
    outFile.write(text + ',' + str(sentementVal) + "\n")
    return


#this function takes in a single tweet and subs in the tokens for user @s and urls
def cleanLine(line):
    noAt = re.sub(r'@[A-Za-z0-9_]+', constants.USER_AT_TOKEN, line)
    url_pattern = re.compile(r'https?://\S+|www\.\S+|t\.co/\S+')
    nohttpsURL = re.sub(url_pattern, '', noAt)
    url_pattern = re.compile(r'http?://\S+|www\.\S+|t\.co/\S+')
    nohttpURL = re.sub(url_pattern, '', nohttpsURL)
    return nohttpURL

#Used to parse its respective file
def parseSentiment_Analysis():
    infile = open(constants.RAW_DATA_PATH + "Sentiment_Analysis.csv")
    i = 0
    for line in infile:
        if i == 0:
            i += 1
            continue
        # removing the newline at the end of the line
        newline = line.replace("\n", "")
        splitArr = newline.split(',')
        cleanedLine = splitArr[3].replace("\"", "")
        cleanedLine = cleanedLine.lower()
        cleanedLine = cleanLine(cleanedLine)

        sentiment = 0
        match splitArr[1].replace("\"", ""):
            case "neutral":
                sentiment = 0
            case "worry":
                sentiment = -1
            case "happiness":
                sentiment = 1
            case "sadness":
                sentiment = -1
            case "love":
                sentiment = 1

        combinedPrintLine(cleanedLine, sentiment, "Sentiment_Analysis.csv")

#Used to parse its respective file
def parseSentiment140():
    infile = open(constants.RAW_DATA_PATH + "training.1600000.processed.noemoticon.csv")
    for line in infile:
        newline = line.replace("\n", "")
        splitArr = newline.split(",")
        cleanedLine = splitArr[5].replace("\"", "")
        cleanedLine = cleanedLine.lower()
        cleanedLine = cleanLine(cleanedLine)
        sentiment = 0
        match splitArr[0]:
            case "2":
                sentiment = 0
            case "4":
                sentiment = 1
            case "0":
                sentiment = -1
        print(cleanedLine)
        combinedPrintLine(cleanedLine, sentiment, "sentiment140.csv")
#Used to parse its respective file
def parseTweets():
    infile = open(constants.RAW_DATA_PATH + "Tweets.csv")
    i = 0
    for line in infile:
        if i == 0:
            i += 1
            continue
        newline = line.replace("\n", "")
        splitArr = newline.split(",")
        cleanedLine = splitArr[1].replace("\"", "")
        cleanedLine = cleanedLine.lower()
        cleanedLine = cleanLine(cleanedLine)
        sentiment = 0
        match splitArr[-1]:
            case "neutral":
                sentiment = 0
            case "positive":
                sentiment = 1
            case "negative":
                sentiment = -1
        print(cleanedLine)
        combinedPrintLine(cleanedLine, sentiment, "Tweets.csv")

#Used to parse its respective file
def parseSmile():
    infile = open(constants.RAW_DATA_PATH + "smile.csv")
    
    for line in infile:
        newline = line.replace("\n", "")
        splitArr = newline.split(",")
        cleanedLine = splitArr[0].replace("\"", "")
        cleanedLine = cleanedLine.lower()
        cleanedLine = cleanLine(cleanedLine)
        sentiment = 0
        match splitArr[-1]:
            case 0:
                sentiment = 0
            case 1:
                sentiment = 1
            case -1:
                sentiment = -1
        print(cleanedLine)
        combinedPrintLine(cleanedLine, sentiment, "smile.csv")



#used to take in a file and split it into 80/10/10 train/test/validation files
def splitData(file, lines):
    inFile = open(constants.CLEANED_DATA_PATH + file, "r")
    testFile = open(constants.FINAL_DATA_PATH + "test.csv", "a")
    # testFile = open(constants.FINAL_DATA_PATH + "test.csv", "x")
    valFile = open(constants.FINAL_DATA_PATH + "val.csv", "a")
    # valFile = open(constants.FINAL_DATA_PATH + "val.csv", "x")
    trainFile = open(constants.FINAL_DATA_PATH + "train.csv", "a")
    # trainFile = open(constants.FINAL_DATA_PATH + "train.csv", "x")

    testAndValLines = math.floor(lines * .1)
    trainLines = lines - (testAndValLines * 2)

    i = 0
    while i != testAndValLines:
        testLine = inFile.readline()
        valLine = inFile.readline()

        testFile.write(testLine)
        valFile.write(valLine)

        i += 1

    i = 0
    while i != trainLines:
        trainLine = inFile.readline()

        trainFile.write(trainLine)

        i += 1
    return


# parseTweets()
# parseSentiment140()
# parseTrain()
# parseSentiment_Analysis()
parseSmile()

# splitData("Tweets.csv", 47480)
# splitData("Sentiment_Analysis.csv", 40000)
# splitData("sentiment140.csv", 1600000)