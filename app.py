from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
knn_model = pickle.load(open('knn_model.pkl','rb'))

# Create ID to name dictionary
def read_item_names():
    file_name = "data/id_to_name.csv"
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, encoding="ISO-8859-1") as f:
        # skip header line
        next(f)
        for line in f:
            line = line.rstrip()
            line = line.split(",")
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

rid_to_name, name_to_rid = read_item_names()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    print("Prediction Request")
    """Grabs the input values and uses them to make prediction"""
    anime_name = request.form["anime_name"]
    try:    
        raw_id = name_to_rid[anime_name]
    except:
        print("Invalid Name")
        return render_template('index.html', prediction_text=f'Invalid Anime Name')
    
    predictions = knn_model.get_neighbors(int(raw_id), k=10)  # this returns a list e.g. [127.20488798], so pick first element [0]
    return render_template('index.html', prediction_text=f'{predictions}')
    
    
    

if __name__ == '__main__':
    app.run()