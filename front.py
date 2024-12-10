from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST','GET'])
def upload():
    data = []
    data1 = []
    if request.method == 'POST':
        data.append(request.form.get('aged'))
        data.append(request.form.get('vehicle_type'))
        data.append(request.form.get('agev'))
        data.append(request.form.get('gender'))
        data.append(request.form.get('speed'))
        data1.append(request.form.get('lat'))
        data1.append(request.form.get('long'))
        data1.append(request.form.get('day'))
        data1.append(request.form.get('weather'))
        model = joblib.load('accident.pkl')
        pred = int(model.predict([data])[0])
        if pred == 1:
            pre = f"{pred} - FATAL"
        elif pred == 2:
            pre = f"{pred} - SERIOUS"
        elif pred == 3:
            pre = f"{pred} - SLIGHT"

        return render_template('/index.html',view = 'style=display:block',  value = pre, view1 = 'style=display:none')

if __name__ == '__main__':
    app.run(debug=True)