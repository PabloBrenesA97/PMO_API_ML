"""
  @author Jose Pablo Brenes Alfaro
"""
# Global Libraries
from flask import Flask, request
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask app
app = Flask(__name__)
# Charge model
et_model = load_model('et_model') 
# Global dataset
df = pd.read_csv('final_df.csv')

@app.route('/')
def welcome():
  return "Welcome All"

@app.route('/predict')
def predict_estimation_future():
  """ 
    Method to predict estimation
  """
  # Request args
  name = str(request.args.get('name'))
  hours = str(request.args.get('hours'))

  # Get table with NLP
  input_df = transform_data(name, hours)

  # Predict
  predictions_df = predict_model(estimator=et_model, data=input_df)

  return {'label': predictions_df['Label'][0], 'score': predictions_df['Score'][0]}

def transform_data(name, hours):
  """ 
    Method to build a new input_df with NLP table included into dataframe.
  """

  ## Add new line 
  new_row = {'name': name, 'hours': hours}
  df_proc = df.append(new_row, ignore_index=True)
  
  # Embedding name
  vectorizer_name = TfidfVectorizer()
  data_name = vectorizer_name.fit_transform(df_proc.name)
  tfidf_tokens_name = vectorizer_name.get_feature_names()
  result_df = pd.DataFrame(data = data_name.toarray(),columns = tfidf_tokens_name)
  result_df = result_df.tail(1)

  # Adding hours
  result_df['hours'] = df_proc.tail(1).hours

  # Reset index
  result_df = result_df.reset_index()

  return result_df


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')