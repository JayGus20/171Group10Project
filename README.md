# ECS 171 Group 10 Project: Anime Recommendation System
To use our frontend demo, a few things must be downloaded:
1. **Flask** is required to run the app.
2. The **Surprise** library is necessary for the models used.

Run the following commands to get those packages:\
&emsp;&emsp;`pip install flask`\
&emsp;&emsp;`pip install surprise`

## To run the backend (jupyter notebook) and the frontend (demo):
1. Download the dataset folder called `data` located in the google drive folder:
    https://drive.google.com/drive/folders/13dV22aqdkK1Hmp-DCSfI2Sv7acGGVz0k?usp=sharing
2. Move the folder into the same directory as the 3 .py files\
a. `app.py`\
b. `finalprojutils.py`\
c. `ProjectCode.ipynb`
3. To run the backend, open the Jupyter notebook `ProjectCode.ipynb` and run the cells sequentially.
4. Make sure that the model files `knn_model.pkl` and `svd_model.pkl` both exist. If they don't, they should be generated at the end of the notebook.
5. To run the frontend, run `app.py` in the terminal by typing *"[Path]/anaconda3/python.exe [Path]/app.py"*
    where *[Path]* is the path to the folder or file specified following. If an IDE such as Visual Studio Code is being used, then the run button should start the app automatically.

## Instructions for the demo:
The KNN model is shown on the left, and the SVD model is shown on the right.\
Make sure the file svd_model.pkl is in the same directory as `app.py`.

### KNN Model instructions:
1. Type in the name of an anime in the textbox. Make sure the name matches what is shown on myanimelist.com.
2. Results are shown in the field below the textbox, which are 10 anime recommendations.

### SVD Model instructions:
1. Type in an anime name followed by a score separated by a comma as shown in the textbox. Each anime and its rating must be on its own line.
2. Results will be shown in the field below with top 25 animes based on scores.