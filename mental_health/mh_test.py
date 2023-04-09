import speech_recognition as sr
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.tokenize import word_tokenize
import pyttsx3
import openai
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# установка API-ключа OpenAI
openai.api_key = "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"

# инициализация синтезатора речи
engine = pyttsx3.init()

# создание объекта распознавания речи
recognizer = sr.Recognizer()

# запись речи с микрофона
with sr.Microphone() as source:
    print("Слушаю вас")
    audio = recognizer.listen(source)
    
# сохранение записи в файл
with open("audio.wav", "wb") as f:
    f.write(audio.get_wav_data())

# распознавание текста из записи
try:
    text = recognizer.recognize_google(audio, language='ru-RU')
except sr.UnknownValueError:
    print("Google Speech Recognition не смог распознать речь")
    text = ""

# обработка текста для автоматической корректировки
# пример использования классификатора kNN из библиотеки sklearn
X_train = [["привет", "как", "дела"],
           ["я", "голоден"],
           ["сколько", "времени", "сейчас"],
           ["помощь", "нужна"],
           ["как", "настроение"],
           ["ты", "как"],
           ["как", "делаешь"],
           ["что", "нового"],
           ["ты", "занят"],
           ["чем", "занимаешься"]]

# преобразуем списки слов в строки
X_train = [' '.join(x) for x in X_train]

y_train = ["привет, как дела?", 
           "я голоден", 
           "сейчас время", 
           "вам нужна помощь?",
           "у меня все хорошо, спасибо. А у вас?", 
           "у меня все отлично, спасибо. А у вас?", 
           "я делаюсь лучше, спасибо. А у вас?", 
           "ничего особенного, а у вас?", 
           "немного занят, а у вас?", 
           "я отвечаю на ваши вопросы!"]

# создание модели "мешка слов"
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)


# создание и обучение классификатора kNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_vectors, y_train)

# получение наиболее вероятного предполагаемого текста
predicted_text = []
for word in text.split():
    try:
        predicted_word = knn.predict([predicted_text + [word]])[0]
        predicted_text.append(predicted_word)
    except:
        predicted_text.append(word)

# вывод текста
final_text = " ".join(predicted_text)
print( final_text)


# токенизация текста на слова
tokens = word_tokenize(final_text)

# анализ тональности текста
sentiment = SentimentIntensityAnalyzer().polarity_scores(final_text)['compound']
# запрос к GPT-3 для генерации ответа
model = "text-davinci-002"
response = openai.Completion.create(engine=model, prompt=final_text, max_tokens=1024)

# получение ответа от OpenAI
answer = response.choices[0].text

# печатаем ответ на экран
print(response.choices[0].text)

# синтез речи на основе ответа
engine.say(answer)

# произношение синтезированной речи
engine.runAndWait()

