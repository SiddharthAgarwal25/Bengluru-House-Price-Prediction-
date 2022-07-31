from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Bengluru Houses Data Cleaned.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)
pipe = pickle.load(open('XgboostModel.pkl', 'rb'))
print(data)

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locs=locations)


@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('sqfeet')
    print(locations, bhk, bath, sqft)
    inputs = pd.DataFrame([[locations, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(inputs)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)