from flask import Flask, request, jsonify, render_template
import nltk
from nltk.stem import WordNetLemmatizer, RSLPStemmer
import json
import random
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('rslp')  # Lematizador para português
nltk.download('stopwords')


app = Flask(__name__)


# Inicializar o lemmatizer
stemmer = RSLPStemmer()
stop_words = set(stopwords.words('portuguese'))
lemmatizer = WordNetLemmatizer()

# Carregar os dados do bot
with open('intents.json') as json_data:
    intents = json.load(json_data)


def clean_up_sentence(sentence):
    # Tokenizar e normalizar a frase
    sentence_words = nltk.word_tokenize(sentence, language='portuguese')
    sentence_words = [word.lower() for word in sentence_words if word.lower() not in stop_words]
    return sentence_words

def predict_class(sentence):
    words = clean_up_sentence(sentence)
    print(f"Cleaned words: {words}")
    results = []
    
    for intent in intents['intents']:
        tag = intent['tag']
        pattern_words = [word.lower() for pattern in intent['patterns'] for word in nltk.word_tokenize(pattern, language='portuguese')]
        print(f"Pattern words for tag '{tag}': {pattern_words}")
        if any(word in pattern_words for word in words):
            results.append({"intent": tag, "probability": 1})
    
    if not results:
        return None
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    print(f"Predicted intent: {results[0]['intent']}")
    return results[0]['intent']


def get_response(intent, intents_json):
    tag = intent
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print(f"Received message: {data}")  # Verificar o que está sendo recebido
    user_message = data['message']
    predicted_intent = predict_class(user_message)
    
    if predicted_intent is None:
        response = "Desculpe, não entendi sua pergunta. Por favor, tente reformulá-la."
    else:
        response = get_response(predicted_intent, intents)
    
    print(f"Sending response: {response}")  # Verificar a resposta gerada
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
