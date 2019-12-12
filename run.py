from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from model.Model import ModelBuilder
import numpy as np

app=Flask(__name__)

#MAIN OBJECT: object that contains "predict()" method and other useful methods
build_model         =   ModelBuilder(word2vec_path="./model/Word2Vec/frWac_non_lem_no_postag_no_phrase_500_skip_cut200.bin") 

#LOADING THE MODEL
categories_classification_model = load_model("./model/ML_models/rental_incidents_classification_model.h5")

#LOADING VARIABLES
tf_idf_dict         =   build_model.load_variables(path="./model/saved_variables/tf_idf_dict.pickle")
categories          =   build_model.load_variables(path="./model/saved_variables/categories.pickle")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def submitText():  
    #PREDICTION FOR A NEW OBSERVATION
    label                                   =   None
    rounded_probability                     =   None
    sentence                                =   request.form["message"]
    if len(sentence.strip())!=0:
        X_new = build_model.get_only_meaningful_sentences_without_label(tf_idf_dict, [sentence])
        if len(X_new)==0:
        	label 		= "🔴 Texte non pertinent"
        	rounded_probability	= 0
        else:
    	    y_new_obs_predicted, prediction_details = 	build_model.predict(X_new, categories_classification_model, categories) 
    	    label 									= 	y_new_obs_predicted[0][0] 
    	    probability 							= 	y_new_obs_predicted[0][1]
    	    rounded_probability                     =  	np.round(float(probability),2)
        return jsonify({'status':'OK', 'label':label, 'probability':rounded_probability})
    return jsonify({'status':'OK', 'warning':" ⚠️ Vous n'avez rien entré"})

if __name__ == "__main__":
    app.run(debug=True)
    







