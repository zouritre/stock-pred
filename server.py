from flask import Flask, render_template, request, send_from_directory
import numpy as np
from flask_socketio import SocketIO, emit
from distutils.util import strtobool

import stock_predictionDL as sp


settings = np.array([{"datasetname": "DatasetName", "retrain": "False", "forecast": "1"}])

datasetname = settings[0]["datasetname"]
retrain = settings[0]["retrain"]
forecast = settings[0]["forecast"]

app = Flask(__name__, template_folder="static/html")
app.config['CORS_HEADERS'] = 'Content-Type'
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on('connect')
def connect():
    print("Client connected")


@socketio.on('submit')
def dispdata(data):
    datas = sp.getOHLC(data["datasetname"], bool(strtobool(data['retrainmodel'])), int(data['forecastvalue']), data["timeframe"])
    emit("forecast", {"data": datas})
@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")


# @app.route("/submit", methods=['POST'])
# def submit():
#     settings[0]["retrain"] = 'False'
#     for elem in request.form:
#         settings[0][elem] = request.form[elem]
#     print(settings)
#     resp = app.make_response("")
#     resp.headers['status'] = 200
#     return resp


if __name__ == "__main__":
    # app.run(debug=True)
    socketio.run(app, debug=True)

    # forcast = sp.getOHLC("Binance_BTCUSDT_1h.csv", False, 3)
