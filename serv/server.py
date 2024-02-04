from flask import Flask, jsonify
from flask_cors import CORS
from poem_search import *

app = Flask(__name__)
CORS(app, resources={r"/process_data": {"origins": "*"}})

@app.route('/process_data', methods=['POST'])
def process_data():
    result_data = search_poems()
    return jsonify(result_data)

@app.route("/")
def index():
    return "<h1>Hello!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)

