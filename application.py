from flask import Flask, jsonify,render_template,request
import pickle , numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application

# mimport ridge_model 
ridge_model=pickle.load(open("lassocv_model.pkl","rb"))
sascaler=pickle.load(open("scaler.pkl","rb"))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature=float(request.form.get("temperature"))
        RH=float(request.form.get("rh"))
        WS=float(request.form.get("ws"))
        rain=float(request.form.get("rain"))
        ffmc=float(request.form.get("ffmc"))
        dmc=float(request.form.get("dmc"))
        isi=float(request.form.get("isi"))
        bui=float(request.form.get("bui"))
        Classes=float(request.form.get("classes"))
        Region=float(request.form.get("region"))
        input_data=np.array([[temperature,RH,WS,rain,ffmc,dmc,isi,bui,Classes,Region]])
        std_data=sascaler.transform(input_data)
        fwi_pred=ridge_model.predict(std_data)
        return render_template("home.html",result=round(fwi_pred[0],2))
    else:
        return render_template("home.html")



def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) 