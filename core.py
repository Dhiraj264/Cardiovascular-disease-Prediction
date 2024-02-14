import flask
from flask import render_template
from flask import request

import numpy as np
from flask import render_template
import pickle


app = flask.Flask(__name__,template_folder= "templates")


@app.route('/')
def index():
    return render_template('mainpage.html')


@app.route('/act', methods=['POST', 'GET'])
def index1():
    file = request.form
    l = file.values()
    l = list(l)
    print(file)
    print(*l)

    inp_arr = np.asarray(l,dtype=np.float64)
    i = inp_arr.reshape(1, -1)
    

    pick1 = open(r"studentmodel.pkl", "rb")
    pick2 = open(r"randmodel.pkl", "rb")
    pick3 = open(r"decmodel.pkl", "rb")
    
    # with open('studentmodel.pkl', 'rb') as file: 
    #     log = pickle.load(file)
    # with open('randmodel.pkl', 'rb') as file: 
    #     dec = pickle.load(file) 
    # with open('decmodel.pkl', 'rb') as file: 
    #     rnd = pickle.load(file)  


    log = pickle.load(pick1)
    dec = pickle.load(pick3)
    rnd = pickle.load(pick2)

    prediction1 = log.predict(i)
    prediction2 = dec.predict(i)
    prediction3 = rnd.predict(i)
    lst = [0, 0, 0]
    lst[0] = prediction1[0]
    lst[1] = prediction2[0]
    lst[2] = prediction3[0]
    print(*lst)

    zero = 0
    one = 0
    for result in lst:
        if result == 0:
            zero += 1
        elif result == 1:
            one += 1
    print(zero)
    print(one)
    if zero > one:
        z = 0
    else:
        z = 1
    if z == 1:
        return render_template("results.html", b=z)
    else:
        return render_template("yes.html", b=z)
if __name__ ==  "__main__":
    app.run(debug = True,port = 5000)