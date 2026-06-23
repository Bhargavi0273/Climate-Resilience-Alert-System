from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    temp = request.form["temperature"]
    rainfall = request.form["rainfall"]
    humidity = request.form["humidity"]
    wind = request.form["wind_speed"]

    prediction = model.predict([[
        float(temp),
        float(rainfall),
        float(humidity),
        float(wind)
    ]])

    result = encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        prediction_text=f"Climate Risk: {result}",
        temperature=temp,
        rainfall=rainfall,
        humidity=humidity,
        wind_speed=wind
    )

if __name__ == "__main__":
    app.run(debug=True)