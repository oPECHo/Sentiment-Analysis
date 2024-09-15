import pandas as pd
from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from function import split_text
import joblib

# โหลดข้อมูลจาก URL ลงใน DataFrame
df = pd.read_csv(
    "https://raw.githubusercontent.com/PyThaiNLP/thai-sentiment-analysis-dataset/master/review_shopping.csv",
    sep="\t",
    names=["text", "sentiment"],
    header=None,
)

# โหลด stopwords ภาษาไทย
thai_stopwords = list(thai_stopwords())


def text_process(text):
    """
    ประมวลผลข้อความโดย:
    1. ลบเครื่องหมายวรรคตอนและอักขระพิเศษ
    2. แบ่งข้อความเป็นคำ (tokenize)
    3. ลบคำที่เป็น stopwords
    """

    # ลบเครื่องหมายวรรคตอนและอักขระพิเศษ
    final = "".join(
        u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ")
    )

    # แบ่งข้อความเป็นคำ
    final = word_tokenize(final)

    # รวมคำที่ถูกแยกออกมาเป็นสตริงเดียว
    final = " ".join(word for word in final)

    # ลบคำที่เป็น stopwords (แปลงคำเป็นตัวพิมพ์เล็กก่อนตรวจสอบกับ stopwords)
    final = " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final


# ใช้ฟังก์ชัน text_process กับคอลัมน์ 'text' และเก็บผลลัพธ์ในคอลัมน์ใหม่ 'text_tokens'
df["text_tokens"] = df["text"].apply(text_process)

# เตรียมข้อมูลสำหรับการฝึกสอนและการทดสอบ
X = df[["text_tokens"]]  # คุณลักษณะ (ข้อความที่ประมวลผลแล้ว)
y = df["sentiment"]  # เป้าหมาย (ป้ายกำกับอารมณ์)

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ (70% สำหรับการฝึกสอน, 30% สำหรับการทดสอบ)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)


# เริ่มต้นและฝึกฝน CountVectorizer
# CountVectorizer แปลงข้อความเป็นเมทริกซ์ของการนับคำ
cvec = CountVectorizer(analyzer=split_text)  # ตัวแบ่งคำที่กำหนดเองซึ่งแยกตามช่องว่าง
cvec.fit_transform(X_train["text_tokens"])  # ฝึกฝน vectorizer บนข้อมูลชุดฝึกสอน

# แปลงข้อมูลฝึกสอนเป็นการแสดงผลแบบ bag-of-words
train_bow = cvec.transform(X_train["text_tokens"])

# เริ่มต้นและฝึกฝนโมเดล Logistic Regression
lr = LogisticRegression()
lr.fit(train_bow, y_train)  # ฝึกฝนโมเดลด้วยการแสดงผลแบบ bag-of-words ของข้อมูลฝึกสอน

# แปลงข้อมูลทดสอบเป็นการแสดงผลแบบ bag-of-words
test_bow = cvec.transform(X_test["text_tokens"])

# ทำนายอารมณ์สำหรับข้อมูลทดสอบ
test_predictions = lr.predict(test_bow)

# แสดงรายงานการจำแนกประเภทเพื่อประเมินประสิทธิภาพของโมเดล
# รายงานการจำแนกประเภทรวมถึงเมตริกเช่น ความแม่นยำ, การเรียกคืน, และ F1-score
print("Test Data Metrics:")
print(f"Accuracy: {accuracy_score(test_predictions, y_test)}")
print(classification_report(test_predictions, y_test))

joblib.dump(lr, "model/sentiment_model.pkl")
joblib.dump(cvec, "model/vectorizer.pkl")

print("Save Model Success")

# # ฟังก์ชันเพื่อทำนายอารมณ์จากข้อมูลที่ป้อนโดยผู้ใช้
# def predict_sentiment(text):
#     """
#     ทำนายอารมณ์ของข้อความที่ป้อนโดย:
#     1. ประมวลผลข้อความ
#     2. แปลงข้อความโดยใช้ CountVectorizer ที่ฝึกฝนแล้ว
#     3. ทำนายอารมณ์โดยใช้โมเดล Logistic Regression ที่ฝึกฝนแล้ว
#     """
#     processed_text = text_process(text) # ประมวลผลข้อความ
#     text_bow = cvec.transform(pd.Series([processed_text])) # แปลงข้อความที่ประมวลผลแล้ว
#     prediction = lr.predict(text_bow) # ทำนายอารมณ์
#     return prediction[0] # ส่งคืนอารมณ์ที่ทำนาย

# # รับข้อมูลจากผู้ใช้และทำนายอารมณ์จนกว่าผู้ใช้จะพิมพ์ 'exit'
# while True:
#     user_input = input("Enter text for sentiment analysis (or type 'exit' to quit): ")
#     if user_input.lower() == 'exit': # ออกจากลูปหากผู้ใช้พิมพ์ 'exit'
#         break
#     sentiment = predict_sentiment(user_input) # ทำนายอารมณ์ของข้อมูลที่ป้อนโดยผู้ใช้
#     print(f"The predicted sentiment is: {sentiment}") # แสดงอารมณ์ที่ทำนาย
