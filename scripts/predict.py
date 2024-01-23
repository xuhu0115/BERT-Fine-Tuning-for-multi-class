import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = "cpu"

# 加载微调后的模型和分词器
model = BertForSequenceClassification.from_pretrained("project/littool/models/bert_multiclass_model")
tokenizer = BertTokenizer.from_pretrained("project/littool/bert-base-uncased")

# 设置模型为评估模式
model.eval()

# 输入文本进行预测
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities[0].tolist()

# 例子
text_to_predict = "Across a few prohibitive miles: The impact of the Anti-Poverty Relocation Program in China"
predicted_class, probabilities = predict(text_to_predict)

print(f"Predicted Class: {predicted_class}")
print(f"Probabilities: {probabilities}")
