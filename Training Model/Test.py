import joblib
import pandas as pd
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from sklearn.metrics import accuracy_score, classification_report

thai_stopwords = list(thai_stopwords())


def text_process(text):
    final = "".join(
        u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ")
    )
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(
        word for word in final.split() if word.lower() not in thai_stopwords
    )
    return final


# โหลดโมเดลและเวกเตอร์
lr = joblib.load("model/sentiment_model.pkl")
cvec = joblib.load("model/vectorizer.pkl")

# โหลดชุดข้อมูลทดสอบ (คุณอาจจะต้องโหลดชุดข้อมูลใหม่)
df = pd.read_csv(
    "https://raw.githubusercontent.com/PyThaiNLP/thai-sentiment-analysis-dataset/master/review_shopping.csv",
    sep="\t",
    names=["text", "sentiment"],
    header=None,
)
df["text_tokens"] = df["text"].apply(text_process)

X_test = df[["text_tokens"]]
y_test = df["sentiment"]

test_bow = cvec.transform(X_test["text_tokens"])
test_predictions = lr.predict(test_bow)

print("Test Data Metrics:")
print(f"Accuracy: {accuracy_score(test_predictions, y_test)}")
print(classification_report(test_predictions, y_test))


def predict_sentiment(text):
    processed_text = text_process(text)
    text_bow = cvec.transform(pd.Series([processed_text]))
    prediction = lr.predict(text_bow)
    return prediction[0]


while True:
    user_input = input("Enter text for sentiment analysis (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    sentiment = predict_sentiment(user_input)
    print(f"The predicted sentiment is: {sentiment}")
