# VaderBaseline.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Mapping to label spce:
#   0 -> negative
#   1 -> neutrual
#   2 -> positive
class VaderBaseline:
    """
     neg_threshold and pos_threshold are the standard VADER defaults:
        negative: compound <= -0.05
        netural: -0.05 < compound < 0.05 
        positive: compound > =0.05
    """
    def __int__(self, neg_threshold: float = -0.05, pos_threshold: float = 0.05):
        self.analyzer = SentimentIntensityAnalyzer()
        self.neg_threshold = neg_threshold
        self.pos_threshold = pos_threshold

    def predict_label(self, text: str) -> int:
        """
         Return a single label: 0(negative), 1(netural), 2(positive)
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= self.pos_threshold:
            return 2
        elif compound <= self.neg_threshold:
            return 0
        else:
            return 1 
    
    """
    predict a batch of texts. Return a list of labels 0/1/2
    """
    def predict_batch(self, texts):
        return [self.predict_label(t) for t in texts]

