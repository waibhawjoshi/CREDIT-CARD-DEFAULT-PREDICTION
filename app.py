from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("creditcard_default.html")
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('creditcard_default.html',pred='Customer will default.\n Probability of Customer default is {}'.format(output),bhai="Alert on customer?")
    else:
        return render_template('creditcard_default.html',pred='Customer will not default.\n Probability of Customer default is {}'.format(output),bhai="Your default status is Safe.")

if __name__ == '__main__':
    app.run(debug=True)