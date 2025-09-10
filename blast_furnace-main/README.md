# Blast Furnace Skin Temperature Prediction

This project uses machine learning to predict blast furnace skin temperatures, providing sequential hourly forecasts. It’s implemented with a Flask web app for interactive temperature input and prediction.

## Project Structure

- **`app.py`**  
  The main Flask application file that loads the trained models and provides routes for user input and prediction.
  
- **`models.py`**  
  Includes code for data processing, model training, and evaluation, covering preprocessing steps like feature scaling and model selection (e.g., XGBoost).

- **`templates/`**  
  Contains HTML templates:
  - `index.html`: User input page for temperature values.
  - `result.html`: Displays the predicted temperatures and a line graph of predictions.

- **`static/`**  
  Stores CSS, JavaScript, and any images or generated graphs.

- **`models/`**  
  Holds serialized models (`model_1.pkl`, `model_2.pkl`, etc.) trained for multi-hour temperature predictions.

  ## ScreenShots
  
![Screenshot 2024-10-31 at 1 50 43 PM](https://github.com/user-attachments/assets/c91686e5-fcf7-4f03-83dd-7f3eaa0d563c)
![Screenshot 2024-10-31 at 1 50 53 PM](https://github.com/user-attachments/assets/a52a20e9-1781-4a87-be00-e85dc541c0db)
<img width="864" alt="Screenshot 2024-10-31 at 1 51 00 PM" src="https://github.com/user-attachments/assets/876f6742-b63c-40e2-90ad-ddbba0d2be82">
