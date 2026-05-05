from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Capturing temporal and environmental features
            data = CustomData(
                season=request.form.get('season'),
                hour=request.form.get('hour'), # Restored Time section
                holiday=request.form.get('holiday'),
                workingday=request.form.get('workingday'),
                weather=request.form.get('weather'),
                temp=request.form.get('temp'),
                humidity=request.form.get('humidity'),
                windspeed=request.form.get('windspeed')
            )

            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # UPDATED: Convert the prediction (e.g., 187.5) to a whole number (187)
            final_result = int(results[0]) 
            
            return render_template('home.html', results=final_result)
        
        except Exception as e:
            # Safely displays the "Unknown Category" error on the page
            return render_template('home.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)