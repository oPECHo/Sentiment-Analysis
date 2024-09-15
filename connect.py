from flask import Flask, request, abort
import requests
import json
import pandas as pd
import joblib
from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from sklearn.metrics import classification_report, accuracy_score
from function import split_text

app = Flask(__name__)

# Thai stopwords
thai_stopwords = list(thai_stopwords())

lr = joblib.load("Training Model/model/sentiment_model.pkl")
cvec = joblib.load("Training Model/model/vectorizer.pkl")


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


# Define sentiment prediction function
def predict_sentiment(text):
    tokens = text_process(text)
    bow = cvec.transform(pd.Series([tokens]))
    prediction = lr.predict(bow)
    return prediction[0]


@app.route("/", methods=["POST", "GET"])
def webhook():
    if request.method == "POST":
        payload = request.json
        reply_token = payload["events"][0]["replyToken"]
        message = payload["events"][0]["message"]["text"]

        if message.lower() == "สถานะ":
            # Print metrics to console
            print("Test Data Metrics:")
            print(f"Accuracy: {accuracy_score(test_predictions, y_test)}")
            print(classification_report(test_predictions, y_test))
            reply_message = f"Accuracy: {accuracy_score(test_predictions, y_test)} \nข้อมูลการทดสอบอื่นถูกพิมพ์ลงใน console แล้ว"
        else:
            print(f"Received message: {message}")
            sentiment = predict_sentiment(message)
            if sentiment == "pos":
                reply_message = f"ข้อความข้างต้นแสดงอารมณ์เชิงบวก"
            elif sentiment == "neg":
                reply_message = f"ข้อความข้างต้นแสดงอารมณ์เชิงลบ"

        reply_message_to_line(
            reply_token,
            reply_message,
            "[Channel_access_token]",
        )  # Channel access token

        return json.dumps({"status": "success"}), 200
    else:
        abort(400)


def reply_message_to_line(reply_token, text_message, line_access_token):
    LINE_API = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": f"Bearer {line_access_token}",
    }
    data = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text_message}],
    }
    response = requests.post(LINE_API, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")


if __name__ == "__main__":
    app.run(debug=True)
