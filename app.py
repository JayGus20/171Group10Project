from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
knn_model = pickle.load(open('knn_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    rooms = int(request.form["rooms"])
    distance = int(request.form["distance"])
    prediction = knn_model.predict([[rooms, distance]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    recommendations = round(prediction[0], 2) 

if __name__ == '__main__':
    app.run()