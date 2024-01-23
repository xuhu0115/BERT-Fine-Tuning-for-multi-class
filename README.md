# BERT-Fine-Tuning-for-multi-class

## clone the project
```bash
git clone git clone git@github.com:xuhu0115/BERT-Fine-Tuning-for-multi-class.git
```

## Modify path
Opening the `predict.py`，go to the following line of code and modify your path:

```python
# 加载微调后的模型和分词器
model = BertForSequenceClassification.from_pretrained("project/littool/models/bert_multiclass_model")
tokenizer = BertTokenizer.from_pretrained("project/littool/bert-base-uncased")

```

## run the predict
```python
python predict.py
```
