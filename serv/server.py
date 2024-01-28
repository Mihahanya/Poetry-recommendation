from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/process_data": {"origins": "*"}})

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json 
    result = {"status": "success", "data": data}
    return jsonify(result)

@app.route("/")
def index():
    return "<h1>Hello!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)

