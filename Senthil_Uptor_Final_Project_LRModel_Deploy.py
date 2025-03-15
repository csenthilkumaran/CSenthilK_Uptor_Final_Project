from flask import request, jsonify, Flask
import pickle

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    with open("Senthil_Final_linear_model_pickling.pkl","rb") as obj:
        LRmodel = pickle.load(obj)

# Get Input data as Json
    data=request.get_json()
    features = data[['Temperature', 'Vibration', 'PowerConsumption', 'NetworkLatency','PacketLoss']]  # Expecting input as {"features": [[value1],[value2]...]}
#
    predictions = LRmodel.predict([features])
    return jsonify({"predictions":predictions.tolist()})

if __name__ == "__main__":
    app.run(debug= True)