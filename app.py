from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
knn_model = pickle.load(open('knn_model.pkl','rb'))
svd_model = pickle.load(open('svd_model.pkl','rb'))

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
recommendation_count = 10

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    print("KNN Prediction Request")
    """Grabs the input values and uses them to make prediction"""

    anime_name = request.form["anime_name_knn"]
    print(anime_name)

    try:    
        raw_id = name_to_rid[anime_name]
    except:
        print("Invalid Name")
        return render_template('index.html', prediction_text=f'Invalid Anime Name')
    
    prediction_ids = knn_model.get_neighbors(int(raw_id), k=recommendation_count)  # this returns a list e.g. [127.20488798], so pick first element [0]
    prediction_ids = list((knn_model.trainset.to_raw_iid(inner_id) for inner_id in prediction_ids))
    prediction_names = [rid_to_name[str(rid)] for rid in prediction_ids]
    
    predictions = ""
    for i in range(recommendation_count):
        predictions += prediction_names[i] + " (" + str(prediction_ids[i]) + ")\n"
    return render_template('index.html', prediction_knn=f'{predictions}')

@app.route('/predict2',methods=['POST'])
def predict2():
    print("SVD Prediction Request")
    """Grabs the input values and uses them to make prediction"""
    anime_name = request.form["anime_name_svd"]

    item_count = 
    item_factors = svd_model.qi

    return render_template('index.html', prediction_svd=f'{predictions}')
    
if __name__ == '__main__':
    app.run()