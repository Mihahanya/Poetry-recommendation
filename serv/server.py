from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from poem_search import search_poems

df = pd.read_pickle('D:/PROJECTS/Poetry-recommendation/poetry_data_prepared.pkl')

app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/process_data": {"origins": "*"}})

@app.route('/poem_request', methods=['POST'])
def process_data():
    request_data = request.json
    result_data = search_poems(request_data['request_text'], request_data['top_n'], request_data['search_priority'])
    return jsonify(result_data)
    

@app.route("/")
def index():
    return "<h1>Hello!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)

