import openai
import speech_recognition  # распознавание пользовательской речи (Speech-To-Text)
import pyttsx3  # синтез речи (Text-To-Speech)
import traceback  # вывод traceback без остановки работы программы при отлове исключений
import json  # работа с json-файлами и json-строками
from termcolor import colored  # вывод цветных логов (для выделения распознанной речи)
import random

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer

import os  # работа с файловой системой


# Установка API-ключа OpenAI
openai.api_key = "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"



class OwnerPerson:
    """
    Информация о владельце, включающие имя, город проживания, родной язык речи, изучаемый язык (для переводов текста)
    """
    name = ""
    home_city = ""
    native_language = ""
    target_language = ""


class VoiceAssistant:
    """
    Настройки голосового ассистента, включающие имя, пол, язык речи
    Примечание: для мультиязычных голосовых ассистентов лучше создать отдельный класс,
    который будет брать перевод из JSON-файла с нужным языком
    """
    name = ""
    sex = ""
    speech_language = ""
    recognition_language = ""

def setup_assistant_voice():
    """
    Установка голоса по умолчанию (индекс может меняться в зависимости от настроек операционной системы)
    """
    voices = ttsEngine.getProperty("voices")

    if assistant.speech_language == "en":
        assistant.recognition_language = "en-US"
        if assistant.sex == "female":
            # Microsoft Zira Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[1].id)
        else:
            # Microsoft David Desktop - English (United States)
            ttsEngine.setProperty("voice", voices[2].id)
    else:
        assistant.recognition_language = "ru-RU"
        # Microsoft Irina Desktop - Russian
        ttsEngine.setProperty("voice", voices[0].id)

# Создание объекта Recognizer и Microphone
recognizer = speech_recognition.Recognizer()
microphone = speech_recognition.Microphone()

def record_and_recognize_audio(*args: tuple):
    """
    Запись и распознавание аудио
    """
    with microphone:
        recognized_data = ""

        # запоминание шумов окружения для последующей очистки звука от них
        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Listening...")
            audio = recognizer.listen(microphone, 5, 5)

            with open("microphone-results.wav", "wb") as file:
                file.write(audio.get_wav_data())

        except speech_recognition.WaitTimeoutError:
            play_voice_assistant_speech(translator.get("Can you check if your microphone is on, please?"))
            traceback.print_exc()
            return

        # использование online-распознавания через Google (высокое качество распознавания)
        try:
            print("Started recognition...")
            recognized_data = recognizer.recognize_google(audio, language=assistant.recognition_language).lower()

        except speech_recognition.UnknownValueError:
            pass  # play_voice_assistant_speech("What did you say again?")
        return recognized_data

def play_voice_assistant_speech(answer):
    """
    Проигрывание речи ответов голосового ассистента (без сохранения аудио)
    :param text_to_speech: текст, который нужно преобразовать в речь
    """
    ttsEngine.say(str(answer))
    ttsEngine.runAndWait()

def play_greetings(*args: tuple):
    """
    Проигрывание случайной приветственной речи
    """
    greetings = [
        translator.get("Hello, {}! How can I help you today?").format(person.name),
        translator.get("Good day to you {}! How can I help you today?").format(person.name)
    ]
    play_voice_assistant_speech(greetings)

# Функции для ответов на различные намерения пользователя
def play_greetings(username=None):
    if username is None:
        return "Здравствуйте! Как я могу вам помочь?"
    else:
        return f"Здравствуйте, {username}! Как я могу вам помочь?"

def play_farewell():
    return random.choice(["До свидания!", "Хорошего дня!", "Удачи!", "Всего доброго!"])

def play_thanks():
    return random.choice(["Пожалуйста!", "Не за что!", "Всегда рад помочь!", "Никогда не откажу в помощи!"])

def play_help():
    return "Я могу помочь вам в следующих вопросах: ..."

config = {
    "intents": {
        "greeting": {
            "examples": ["привет", "здравствуй", "добрый день",
                         "hello", "good morning"],
            "responses": play_greetings
        },
        "farewell": {
            "examples": ["до свидания", "пока", "всего доброго",
                         "goodbye", "bye"],
            "responses": play_farewell
        },
        "thanks": {
            "examples": ["спасибо", "благодарю", "thank you",
                         "thanks a lot"],
            "responses": play_thanks
        },
        "help": {
            "examples": ["помощь", "что ты умеешь?", "what can you do?",
                         "помоги мне"],
            "responses": play_help
        }
    }
}

# Define target_vector before it is imported
target_vector = []

def get_intent(request):
    """
    Получение намерения пользователя на основе запроса
    """
    # Загружаем модель и векторизатор
    global classifier, vectorizer, target_vector
    if classifier is None:
        classifier = LinearSVC()
    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    # Подготавливаем данные для обучения
    if len(target_vector) < 2:
        prepare_corpus()

    # Получаем наилучшее совпадение
    best_intent = classifier.predict(vectorizer.transform([request]))[0]
    return best_intent

def prepare_corpus():
    """
    Подготовка модели для угадывания намерения пользователя
    """
    global target_vector
    corpus = []
    for intent_name, intent_data in config["intents"].items():
        for example in intent_data["examples"]:
            corpus.append(example)
            target_vector.append(intent_name)

    training_vector = vectorizer.fit_transform(corpus)

    # Check if the data contains only one class
    unique_classes = set(target_vector)
    if len(unique_classes) == 1:
        print(f"This data contains only one class: {unique_classes}. Cannot train OneClassSVM.")
        return

    # Train the OneClassSVM
    svm = OneClassSVM(gamma='scale', nu=0.01) # Hyperparameters may vary depending on your data
    svm.fit(training_vector)

    classifier_probability.fit(training_vector, target_vector)
    classifier.fit(training_vector, target_vector)
    

# Настройки для создания сессии GPT-3
model_engine = "text-davinci-002"
max_tokens = 1024
stop_sequence = "\n"



def make_preparations():
    """
    Подготовка глобальных переменных к запуску приложения
    """
    global recognizer, microphone, ttsEngine, person, assistant, translator, vectorizer, classifier_probability, classifier

    # инициализация инструментов распознавания и ввода речи
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()

    # инициализация инструмента синтеза речи
    ttsEngine = pyttsx3.init()

    # настройка данных пользователя
    person = OwnerPerson()
    person.name = "Nurdaulet"
    person.home_city = "Aktobe"
    person.native_language = "ru"
    person.target_language = "en"

    # настройка данных голосового помощника
    assistant = VoiceAssistant()
    assistant.name = "Aurora"
    assistant.sex = "female"
    assistant.speech_language = "ru"

    # установка голоса по умолчанию
    setup_assistant_voice()

    # подготовка корпуса для распознавания запросов пользователя с некоторой вероятностью (поиск похожих)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    classifier_probability = LogisticRegression()
    classifier = LinearSVC()
    prepare_corpus()

while True:
   assistant = VoiceAssistant() # создание объекта вашего голосового помощника
   record_and_recognize_audio(assistant) # передача объекта в качестве аргумента
    # Завершение диалога, если пользователь ввел "пока"
   user_input = record_and_recognize_audio()
   if user_input.lower() == "пока":
       print("До свидания!")
       break

        
    # Создание запроса к GPT-3
   prompt = f"User: {record_and_recognize_audio}\nAI:"
   response = openai.Completion.create(
       engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        stop=stop_sequence
    )
    
    # Получение ответа от GPT-3
   answer = response.choices[0].text.strip()
   print(answer)



if __name__ == "__main__":
    make_preparations()

    while True:
        # старт записи речи с последующим выводом распознанной речи и удалением записанного в микрофон аудио
        voice_input = record_and_recognize_audio()

        if os.path.exists("microphone-results.wav"):
            os.remove("microphone-results.wav")

        print(colored(voice_input, "blue"))

        # отделение комманд от дополнительной информации (аргументов)
        if voice_input:
            voice_input_parts = voice_input.split(" ")

            # если было сказано одно слово - выполняем команду сразу без дополнительных аргументов
            if len(voice_input_parts) == 1:
                intent = get_intent(voice_input)
                if intent:
                    config["intents"][intent]["responses"]()
                else:
                    config["failure_phrases"]()

            # в случае длинной фразы - выполняется поиск ключевой фразы и аргументов через каждое слово,
            # пока не будет найдено совпадение
            if len(voice_input_parts) > 1:
                for guess in range(len(voice_input_parts)):
                    intent = get_intent((" ".join(voice_input_parts[0:guess])).strip())
                    print(intent)
                    if intent:
                        command_options = [voice_input_parts[guess:len(voice_input_parts)]]
                        print(command_options)
                        config["intents"][intent]["responses"](*command_options)
                        break
                    if not intent and guess == len(voice_input_parts)-1:
                        config["failure_phrases"]()


