from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Set your Groq API key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Set this in your .env file

def get_caste_prediction(name, surname, place_of_origin):
    """
    Make a call to Groq's LLaMA model to predict caste and government category
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        prompt = f"""
You are a helpful assistant that provides caste and category predictions for Indian surnames for educational and research purposes.

Given the following information:
Name: {name}
Surname: {surname}
Place of Origin: {place_of_origin}

Please respond in this compact format:
Caste: [predicted caste]
Category: [General/OBC/SC/ST]
Brief Info: One or two lines explaining why this prediction was made based on surname and region. Avoid lengthy explanations. This is a prediction and may not be fully accurate.
"""


        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error from Groq API: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error making prediction: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name', '').strip()
        surname = request.form.get('surname', '').strip()
        place_of_origin = request.form.get('place_of_origin', '').strip()

        if not all([name, surname, place_of_origin]):
            return jsonify({'error': 'All fields (name, surname, place of origin) are required'}), 400

        prediction = get_caste_prediction(name, surname, place_of_origin)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'input': {
                'name': name,
                'surname': surname,
                'place_of_origin': place_of_origin
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY environment variable not set!")
        print("Please set it using: export GROQ_API_KEY='your-api-key-here'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
