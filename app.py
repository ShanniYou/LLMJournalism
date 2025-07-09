# By Shanni You, @ 07/06/2025

from flask import Flask, render_template, request, jsonify
from NewsdayChat import generate_response
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    bot_response, sql_source, article_source = generate_response(user_message)
    return jsonify({'response': {
        'final_answer': bot_response, 'sql_source': sql_source, 'article_source': article_source
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
