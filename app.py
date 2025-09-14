from flask import Flask, render_template, request
import pandas as pd
import joblib
import xgboost as xgb

app = Flask(__name__)
model = joblib.load("xgb_model.pkl")

# Mappings
reporting_airline_mapping = {'AA': 0, 'UA': 1, 'DL': 2, 'WN': 3, 'OO': 4}
origin_mapping = {'LAX': 0, 'ORD': 1, 'DFW': 2, 'SFO': 3, 'ATL': 4, 'DEN': 5, 'PHX': 6, 'IAH': 7, 'CLT': 8}
dest_mapping = {'LAX': 0, 'ORD': 1, 'SFO': 2, 'DFW': 3, 'ATL': 4, 'DEN': 5, 'PHX': 6, 'IAH': 7, 'CLT': 8}

feature_columns = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Origin", "Dest", "CRSDepTime",
    "Cancelled", "Diverted", "Distance", "DistanceGroup",
    "ArrDelay", "ArrDelayMinutes", "AirTime"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            data = {
                "Year": int(request.form['Year']),
                "Quarter": int(request.form['Quarter']),
                "Month": int(request.form['Month']),
                "DayofMonth": int(request.form['DayofMonth']),
                "DayOfWeek": int(request.form['DayOfWeek']),
                "Reporting_Airline": reporting_airline_mapping.get(request.form['Reporting_Airline'], -1),
                "Origin": origin_mapping.get(request.form['Origin'], -1),
                "Dest": dest_mapping.get(request.form['Dest'], -1),
                "CRSDepTime": int(request.form['CRSDepTime']),
                "Cancelled": int(request.form['Cancelled']),
                "Diverted": int(request.form['Diverted']),
                "Distance": float(request.form['Distance']),
                "DistanceGroup": int(request.form['DistanceGroup']),
                "ArrDelay": float(request.form['ArrDelay']),
                "ArrDelayMinutes": float(request.form['ArrDelayMinutes']),
                "AirTime": float(request.form['AirTime']),
            }

            if -1 in [data["Reporting_Airline"], data["Origin"], data["Dest"]]:
                result = "Error: Invalid Airline, Origin, or Destination entered!"
            else:
                df = pd.DataFrame([data])
                prediction = model.predict(df)[0]
                result = "Delayed" if prediction == 1 else "On-Time"
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html',
                           result=result,
                           airline_keys=reporting_airline_mapping.keys(),
                           origin_keys=origin_mapping.keys(),
                           dest_keys=dest_mapping.keys())

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000) 
