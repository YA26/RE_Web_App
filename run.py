from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from model.Classifier import IncidentClassifier
import numpy as np

app=Flask(__name__)
app.config['JSON_AS_ASCII'] = False

#MAIN OBJECT: object that contains "predict(), get_json()" method and other useful methods
incident_classifier = IncidentClassifier(Word_2_Vec_path="./model/Word2Vec/frWac_non_lem_no_postag_no_phrase_500_skip_cut200.bin") 

#LOADING THE MODEL
categories_classification_model = load_model("./model/ML_models/rental_incidents_classification_model.h5")

#LOADING VARIABLES
tf_idf_dict = incident_classifier.load_variables(path="./model/saved_variables/tf_idf_dict.pickle")
categories  = incident_classifier.load_variables(path="./model/saved_variables/categories.pickle")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submitForm(): 
    prediction_dict = incident_classifier.get_prediction_json(text=request.form["message"], 
                                                                tf_idf_dict=tf_idf_dict,
                                                                categories=categories, 
                                                                deep_learning_model=categories_classification_model, 
                                                                window_size=4) 

    return jsonify(prediction_dict)
    
@app.route("/get-json/<text>", methods=["GET"])
def submitText(text):  
    prediction_dict = incident_classifier.get_prediction_json(text=text, 
                                                                tf_idf_dict=tf_idf_dict,
                                                                categories=categories,
                                                                deep_learning_model=categories_classification_model, 
                                                                window_size=4) 
    return jsonify(prediction_dict)

"""
if __name__ == "__main__":
    app.run(debug=True)
    
"""






