from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from poem_search import search_poems

app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/process_data": {"origins": "*"}})

@app.route('/poem_request', methods=['POST'])
def process_data():
    request_data = request.json
    print(request_data)

    result_data = search_poems(request_data['request_text'][:1000], request_data['top_n'], request_data['search_priority'])
    
    return jsonify(result_data)
    

@app.route("/")
def index():
    return "<h1>Hello!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
