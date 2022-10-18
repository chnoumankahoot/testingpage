# Load libraries
import flask,requests,json
import pandas as pd
import numpy as np

import keras
from keras.models import load_model
#getting data from Api
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=500')
#converting data from json to Dataframe
hist = pd.DataFrame(json.loads(res.content)['Data'])
#setting time as as index
hist = hist.set_index('time')
#since the datetime is in string we are converting in datetime
hist.index = pd.to_datetime(hist.index, unit='s')
#feature or variable we will use to predict
hist.drop(["conversionType", "conversionSymbol"], axis = 'columns', inplace = True)
target_col = 'close'
# instantiate flask
app = flask.Flask(__name__)

def normalise_zero_base(df):
    return df / df.iloc[0] - 1


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)

    return np.array(window_data)
def denormalise_zero_base(df,val):
    return df['close'].values[:-5] * (val + 1)
# to use it when loading the model


# load the model, and pass in the custom metric function

model = load_model('model.h5')

# define a predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        test2 = extract_window_data(hist.tail(10), window_len=5, zero_base=True)
        futureprice = model.predict(test2).squeeze()

        futureprice = denormalise_zero_base(hist.tail(10),futureprice)
        print("futureprice:",futureprice)
        emptystr=''
        data["predictions"]=emptystr.join(np.array2string(futureprice,separator=','))






    # return a response in json format
    return flask.jsonify(data)

