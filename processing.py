import constants
import re
import pandas as pd


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

        

def parseTrain():
    infile = open(constants.RAW_DATA_PATH + "train.csv")
    i = 0
    for line in infile:
        if i == 0:
            i += 1
            continue
        # removing the newline at the end of the line
        newline = line.replace("\n", "")
        splitArr = newline.split(",")
        cleanedLine = splitArr[1].replace("\"", "")
        cleanedLine = cleanLine(cleanedLine)
        
        sentiment = 0
        match splitArr[-7]:
            case "neutral":
                sentiment = 0
            case "positive":
                sentiment = 1
            case "negative":
                sentiment = -1
        print(cleanedLine)
        combinedPrintLine(cleanedLine, sentiment, "train.csv")

def parseSentiment140():
    infile = open(constants.RAW_DATA_PATH + "training.1600000.processed.noemoticon.csv")
    for line in infile:
        newline = line.replace("\n", "")
        splitArr = newline.split(",")
        cleanedLine = splitArr[5].replace("\"", "")
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






# parseTweets()
# parseSentiment140()
# parseTrain()
# parseSentiment_Analysis()