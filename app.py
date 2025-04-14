from flask import Flask, request, render_template
import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Logger
logger = logging.getLogger("BreastCancerApp")
logger.setLevel(logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            # Collect all 30 features
            features = {key: float(request.form.get(key)) for key in request.form}

            logger.info(f"Received features for prediction: {features}")

            # Prepare data
            data = CustomData(**features)
            df = data.get_data_as_dataframe()

            # Make prediction
            pipeline = PredictPipeline()
            result = pipeline.predict(df)

            outcome = "Malignant" if result[0] == 0 else "Benign"
            return render_template('home.html', results=outcome)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('home.html', results="Prediction failed. Please check input values.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
