from flask import Flask, request, render_template, url_for
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Data/delay.csv")

# Label Encoding
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Features and target
X = df[['year', 'month', 'day', 'carrier', 'origin', 'dest']].values
y = df['delayed']

# Train-test split (optional if model already trained)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=61)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    year = request.form['year']
    month = request.form['month']
    day = request.form['day']
    carrier = request.form['carrier']
    origin = request.form['origin']
    dest = request.form['dest']
    year = int(year)
    month = int(month)
    day = int(day)
    carrier = str(carrier)
    origin = str(origin)
    dest = str(dest)
    
    if year >= 2013:
        x1 = [year,month,day]
        x2 = [carrier, origin, dest]
        x1.extend(x2)
        df1 = pd.DataFrame(data = [x1], columns = ['year', 'month', 'day', 'carrier', 'origin', 'dest'])
        
        df1['carrier'] = le_carrier.transform(df1['carrier'])
        df1['origin'] = le_origin.transform(df1['origin'])
        df1['dest'] = le_dest.transform(df1['dest'])
        
        x = df1[['year', 'month', 'day', 'carrier', 'origin', 'dest']].values
        ans = model.predict(x)
        output = ans
        if output == 1:
            # Delayed → search alternate flight
            encoded_carrier = df1['carrier'].values[0]
            encoded_origin = df1['origin'].values[0]
            encoded_dest = df1['dest'].values[0]

            alternate = df[
                (df['origin'] == encoded_origin) &
                (df['dest'] == encoded_dest) &
                (df['day'] == day) &
                (df['carrier'] != encoded_carrier) &
                (df['delayed'] == 0)
            ].head(1)

            if not alternate.empty:
                alt = alternate.iloc[0]
                suggested_carrier = le_carrier.inverse_transform([int(alt['carrier'])])[0]
                suggested_origin = le_origin.inverse_transform([int(alt['origin'])])[0]
                suggested_dest = le_dest.inverse_transform([int(alt['dest'])])[0]
                p_text = (
                    f"Flight is likely to be delayed. "
                    f"Suggested alternate: {suggested_carrier}, "
                    f"From {suggested_origin} to {suggested_dest} on {int(alt['day'])}/{int(alt['month'])}."
                )
            else:
                p_text = "Flight is likely to be delayed. No suitable alternate found."
        else:
            p_text = "Flight is on time."
    
   


    return render_template('index.html',prediction_text=p_text)

if __name__ == '__main__':
    app.run(debug=False)
