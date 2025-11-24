# Sentiment Analysis Results

Generated: 2025-11-24 17:37

## Overall Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| BERT_Multilingual | 0.7232 | 0.7529 | 0.7166 | 0.7252 |
| Sentiment_BERT | 0.7712 | 0.7974 | 0.7648 | 0.7737 |
| Twitter_RoBERTa | 0.7568 | 0.7728 | 0.7540 | 0.7604 |

## Per-Class Results

### BERT_Multilingual

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.7702 | 0.6703 | 0.7168 |
| Neutral | 0.6287 | 0.8178 | 0.7109 |
| Positive | 0.8599 | 0.6618 | 0.7479 |

### Sentiment_BERT

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.8699 | 0.6865 | 0.7674 |
| Neutral | 0.6746 | 0.8432 | 0.7495 |
| Positive | 0.8478 | 0.7647 | 0.8041 |

### Twitter_RoBERTa

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.8049 | 0.7135 | 0.7564 |
| Neutral | 0.6703 | 0.7839 | 0.7227 |
| Positive | 0.8432 | 0.7647 | 0.8021 |

## Confusion Matrices

### BERT_Multilingual

```
            Pred
         Neg Neu Pos
True Neg  124   54    7
     Neu   28  193   15
     Pos    9   60  135
```

### Sentiment_BERT

```
            Pred
         Neg Neu Pos
True Neg  127   54    4
     Neu   13  199   24
     Pos    6   42  156
```

### Twitter_RoBERTa

```
            Pred
         Neg Neu Pos
True Neg  132   49    4
     Neu   26  185   25
     Pos    6   42  156
```

## Dataset Info

- Training samples: 5000
- Test samples: 625
- Validation samples: 625
- Epochs: 3
