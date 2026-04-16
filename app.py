from flask import Flask,request,render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        Predict_Pipeline=PredictPipeline()
        Predict_Pipeline.Predict(pred_df)

        results = Predict_Pipeline.Predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)

# from flask import Flask, request, render_template
# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# app = Flask(__name__)

# # Home route (optional)
# @app.route('/')
# def home():
#     return "Flask API is running!"

# # JSON API route
# @app.route('/predictjson', methods=['POST'])
# def predict_json():
#     try:
#         data = request.get_json()

#         custom_data = CustomData(
#             gender=data['gender'],
#             race_ethnicity=data['race_ethnicity'],
#             parental_level_of_education=data['parental_level_of_education'],
#             lunch=data['lunch'],
#             test_preparation_course=data['test_preparation_course'],
#             reading_score=float(data['reading_score']),
#             writing_score=float(data['writing_score'])
#         )

#         df = custom_data.get_data_as_dataframe()

#         predict_pipeline = PredictPipeline()
#         result = predict_pipeline.Predict(df)

#         return {"prediction": float(result[0])}

#     except Exception as e:
#         return {"error": str(e)}

# # Run app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)