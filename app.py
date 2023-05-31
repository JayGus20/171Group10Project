from flask import Flask, request, render_template
import pandas as pd
import pickle

from surprise import Reader
from surprise import Dataset

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
            name_to_rid[line[1].lower()] = line[0]

    return rid_to_name, name_to_rid

rid_to_name, name_to_rid = read_item_names()
recommendation_count = 10
#region Default route
@app.route('/')
def home():
    return render_template('index.html')
#endregion

#region KNN route
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
#endregion

#region SVD Route

# Load SVD sample data
svd_samples_df = pd.read_csv('data/frontend_svd_sample.csv')

def create_dataset_from_df(df):
    reader = Reader(rating_scale=(1,10))
    return Dataset.load_from_df(df, reader)

# Appends a new user to the df for refitting and predicting
# We need to do this because suprise does not support iterative training with SVD
def create_predict_dataset(base_df, anime_ids, ratings):
    predictor_df = base_df.copy()
    for i in range(len(anime_ids)):
        predictor_df.loc[len(predictor_df)] = [-1,anime_ids[i], ratings[i]]
    
    return create_dataset_from_df(predictor_df)

# Creates a dataframe for the predictions
def get_predictions(model_instance, user_id, anime_ids):
    # Create prediction set
    reader = Reader(rating_scale=(1, 10))
    predict_data = Dataset.load_from_df(pd.DataFrame({'User ID' : -1, 'Anime ID' : anime_ids, 'Rating' : -1}), reader)

    # Predict
    predictions = model_instance.test(predict_data.build_full_trainset().build_testset())

    return predictions

# Print out information for top N predictions
def get_top_n_string(predictions, n):
    # Sort descending
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    display_string = ""
    display_string += "Top {} predicted scores".format(n)

    # Print information about top n
    ranking = 1
    for prediction in predictions:
        anime_id = prediction.iid
        rating = prediction.est
        name = rid_to_name[str(int(anime_id))]
        display_string += "\n{}. {} ({}) - {}".format(ranking, name, anime_id, round(rating, 4))
        ranking += 1
    
    return display_string

@app.route('/predict2',methods=['POST'])
def predict2():
    print("SVD Prediction Request")
    user_profile_string = request.form["anime_name_svd"]
    
    # Parse input
    anime_ids = []
    ratings = []

    input_lines = user_profile_string.splitlines()
    for line in input_lines:
        try:
            components = line.split('\\')
            name = components[0].strip().lower()
            rating = int(components[1].strip())
            id = name_to_rid[name]

            print(f'{name} ({id}); {rating}')
            anime_ids.append(int(id))
            ratings.append(int(rating))
        except:
            continue
    
    if(len(anime_ids) == 0):
        return render_template('index.html', prediction_svd=f'Enter Ratings one per line in format "anime name \\ rating"', svd_input=user_profile_string)
    
    try:
        # Create dataset
        dataset = create_predict_dataset(svd_samples_df, anime_ids, ratings)

        # Retrain
        print("Fitting SVD")
        svd_model.fit(dataset.build_full_trainset())

        # Generate predictions
        print("Generating Anime IDs")
        predict_anime_ids = list(rid_to_name)

        # Remove user provided ids from being predicted
        predict_anime_ids = [int(id) for id in predict_anime_ids if int(id) not in anime_ids]
        predict_anime_ids = pd.unique(predict_anime_ids)

        print("Getting Predictions")
        predictions = get_predictions(svd_model, -1, predict_anime_ids)

        # Get top N
        result = get_top_n_string(predictions, 25)
        return render_template('index.html', prediction_svd=f'{result}', svd_input=user_profile_string)
    except Exception as exc:
        print(exc)
        return render_template('index.html', prediction_svd=f'Error with model fitting', svd_input=user_profile_string)
#endregion
if __name__ == '__main__':
    app.run()