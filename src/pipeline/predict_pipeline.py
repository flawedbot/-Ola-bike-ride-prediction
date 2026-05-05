import os # FIX: Essential for joining paths
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, season, hour, holiday, workingday, weather, temp, humidity, windspeed):
        self.season = season
        self.hour = hour # RESTORED: Temporal Feature
        self.holiday = holiday
        self.workingday = workingday
        self.weather = weather
        self.temp = temp
        self.humidity = humidity
        self.windspeed = windspeed

    def get_data_as_data_frame(self):
        try:
            # FIX: Mapping Winter (5) back to 4 to prevent model crashes
            input_season = int(self.season)
            if input_season > 4:
                input_season = 4

            custom_data_input_dict = {
                "season": [input_season],
                "hour": [int(self.hour)], 
                "holiday": [int(self.holiday)],
                "workingday": [int(self.workingday)],
                "weather": [int(self.weather)],
                "temp": [float(self.temp)],
                "humidity": [float(self.humidity)],
                "windspeed": [float(self.windspeed)],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)