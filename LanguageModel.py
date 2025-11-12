from transformers import pipeline

class LanguageModel(): 

  def __init__(self, modelSource): 
    self.pipe = pipeline("text-classification", model=modelSource)

  def train(self, data): 
    pass

  def test(self, data): 
    pass