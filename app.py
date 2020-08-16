
import pickle
# Import all the packages you need for your model below
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
# Import Flask for creating API
from flask import Flask, request

# Load the trained model from current directory
with open('./model.pkl', 'rb') as model_pkl:
    knn = pickle.load(model_pkl)

# Initialise a Flask app
app = Flask(__name__)

@app.route("/")                   # at the end point /
def hello():                      # call method hello
    return "Hello World!"         # which returns "hello world"

# Create an API endpoint
@app.route('/predict')
def predict_iris():
    # Read all necessary request parameters
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    pl = request.args.get('pl')
    pw = request.args.get('pw')

    # Use the predict method of the model to 
    # get the prediction for unseen data
    unseen = np.array([[sl, sw, pl, pw]])
    result = knn.predict(unseen)
    
    # return the result back
    return 'Predicted result for observation ' + str(unseen) + ' is: ' + str(result)

#http://localhost:5000/predict?sl=3.2&sw=1.1&pl=1.5&pw=2.1

if __name__ == '__main__':
    app.run(port = '8181')
