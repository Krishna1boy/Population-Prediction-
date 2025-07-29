from flask import *
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the population data
df = pd.read_csv("https://raw.githubusercontent.com/AvantiSakshi/popu/main/pop.csv")    

@app.route('/') 
def hello_world(): 
  return render_template('index.html')
  #return 'Hello, MGCU champs !   ' + str(res[0])
  
# Create a dictionary to store the country models
country_models = {}     
@app.route('/predict', methods=['POST'])
def predict_population():
    country = request.form['country'].lower()
    year = int(request.form['year'])

    # Check if the country model exists
    if country not in country_models:
        # Create a new model for the country
        country_df = df.loc[df['Country Name'].str.lower() == country]
        country_df.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
        country_df = country_df.T
        country_df.dropna(inplace=True)
        country_df = country_df.reset_index()

        x = country_df.iloc[:, 0].values.reshape(-1, 1)
        y = country_df.iloc[:, 1].values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        country_models[country] = model

    # Make the prediction
    model = country_models[country]
    prediction = int(model.coef_[0][0] * year + model.intercept_[0])

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)