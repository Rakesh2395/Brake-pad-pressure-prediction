import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import os

OUTPUT_FOLDER = os.path.join('static', 'output')

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))
model4 = pickle.load(open('model4.pkl', 'rb'))

def plot_contour(x,y,z,resolution = 500,contour_method='linear'):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution),   min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method)
    return X,Y,Z

def predictor(design, app_pressure, pis_radius, pad_thickness, pad_height, pad_width, youngs_mod):

        Inputs = [[app_pressure, pis_radius, pad_thickness, pad_height, pad_width, youngs_mod]]
        results = []
        cordinates = pd.DataFrame()
        cordinates1 = pd.DataFrame()
        cordinates2 = pd.DataFrame()
        cordinates3 = pd.DataFrame()
        cordinates4 = pd.DataFrame()

        if design == 1:
            cordinates01 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design1/Node_cordinates_Design1.csv",
                                     header=None)
            cordinates01.columns = ['Node_Number', 'X_cord', 'Y_cord']
            Model1 = pickle.load(open('model1.pkl', 'rb'))
            results = Model1.predict(Inputs)
            cordinates=cordinates01

        elif design == 2:
            cordinates02 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design2/Node_cordinates_Design2.csv",
                                     header=None)
            cordinates02.columns = ['Node_Number', 'X_cord', 'Y_cord']
            cordinates21 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design2/Node_cordinates_Design2_1.csv",
                                      header=None)
            cordinates21.columns = ['Node_Number', 'X_cord', 'Y_cord']
            cordinates22 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design2/Node_cordinates_Design2_2.csv",
                                      header=None)
            cordinates22.columns = ['Node_Number', 'X_cord', 'Y_cord']
            cordinates23 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design2/Node_cordinates_Design2_3.csv",
                                      header=None)
            cordinates23.columns = ['Node_Number', 'X_cord', 'Y_cord']
            cordinates24 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design2/Node_cordinates_Design2_4.csv",
                                      header=None)
            cordinates24.columns = ['Node_Number', 'X_cord', 'Y_cord']
            Model2 = pickle.load(open('model2.pkl', 'rb'))
            results = Model2.predict(Inputs)
            cordinates=cordinates02
            cordinates1=cordinates21
            cordinates2=cordinates22
            cordinates3=cordinates23
            cordinates4=cordinates24


        elif design == 3:
            cordinates03 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design3/Node_cordinates_Design3.csv",
                                     header=None)
            cordinates03.columns = ['Node_Number', 'X_cord', 'Y_cord']
            cordinates31 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design3/Node_cordinates_Design3_1.csv",
                                      header=None)
            cordinates31.columns = ['Node_Number', 'X_cord', 'Y_cord']

            cordinates32 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design3/Node_cordinates_Design3_2.csv",
                                      header=None)
            cordinates32.columns = ['Node_Number', 'X_cord', 'Y_cord']

            cordinates33 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design3/Node_cordinates_Design3_3.csv",
                                      header=None)
            cordinates33.columns = ['Node_Number', 'X_cord', 'Y_cord']

            cordinates34 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design3/Node_cordinates_Design3_4.csv",
                                      header=None)
            cordinates34.columns = ['Node_Number', 'X_cord', 'Y_cord']
            Model3 = pickle.load(open('model3.pkl', 'rb'))
            results = Model3.predict(Inputs)
            cordinates = cordinates03
            cordinates1 = cordinates31
            cordinates2 = cordinates32
            cordinates3 = cordinates33
            cordinates4 = cordinates34


        elif design == 4:
            cordinates04 = pd.read_csv("C:/Users/sowjanya2014_2/capstone1/design4/Node_cordinates_Design4.csv",
                                     header=None)
            cordinates04.columns = ['Node_Number', 'X_cord', 'Y_cord']
            Model4 = pickle.load(open('model4.pkl', 'rb'))
            results = Model4.predict(Inputs)
            cordinates=cordinates04

        else:
            print("Enter valid design")



        results_df = pd.DataFrame({'Pressure': results[0]})
        results_df["Node_number"] = cordinates["Node_Number"]
        results_df["X_cord"] = cordinates['X_cord']
        results_df["Y_Cord"] = cordinates['Y_cord']

        # For visualization
        x = results_df["X_cord"]
        y = results_df["Y_Cord"]
        z = results_df["Pressure"]

        if design == 2 or design == 3:
            node_list1 = cordinates1["Node_Number"]
            node_list2 = cordinates2["Node_Number"]
            node_list3 = cordinates3["Node_Number"]
            node_list4 = cordinates4["Node_Number"]

            results_1 = results_df[results_df["Node_number"].isin(node_list1)]
            results_2 = results_df[results_df["Node_number"].isin(node_list2)]
            results_3 = results_df[results_df["Node_number"].isin(node_list3)]
            results_4 = results_df[results_df["Node_number"].isin(node_list4)]

            x1 = results_1["X_cord"]
            y1 = results_1["Y_Cord"]
            z1 = results_1["Pressure"]

            x2 = results_2["X_cord"]
            y2 = results_2["Y_Cord"]
            z2 = results_2["Pressure"]

            x3 = results_3["X_cord"]
            y3 = results_3["Y_Cord"]
            z3 = results_3["Pressure"]

            x4 = results_4["X_cord"]
            y4 = results_4["Y_Cord"]
            z4 = results_4["Pressure"]

            X1, Y1, Z1 = plot_contour(x1, y1, z1, resolution=500, contour_method='linear')
            X2, Y2, Z2 = plot_contour(x2, y2, z2, resolution=500, contour_method='linear')
            X3, Y3, Z3 = plot_contour(x3, y3, z3, resolution=500, contour_method='linear')
            X4, Y4, Z4 = plot_contour(x4, y4, z4, resolution=500, contour_method='linear')

            with plt.style.context("classic"):

                fig1, ax1 = plt.subplots(figsize=(5, 5))
                plt.colorbar(ax1.contourf(X1, Y1, Z1), shrink=0.7)
                ax1.xaxis.set_visible(False)
                ax1.yaxis.set_visible(False)

                fig1, ax2 = plt.subplots(figsize=(5, 5))
                plt.colorbar(ax2.contourf(X2, Y2, Z2), shrink=0.7)
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)

                fig1, ax3 = plt.subplots(figsize=(5, 5))
                plt.colorbar(ax3.contourf(X3, Y3, Z3), shrink=0.7)
                ax3.xaxis.set_visible(False)
                ax3.yaxis.set_visible(False)

                fig1, ax4 = plt.subplots(figsize=(5, 5))
                plt.colorbar(ax4.contourf(X4, Y4, Z4), shrink=0.7)
                ax4.xaxis.set_visible(False)
                ax4.yaxis.set_visible(False)

                plt.savefig('C:/Users/sowjanya2014_2/capstone1/deployment/Project1/static/output/plot1.png')


        elif design == 1 or design == 4:
            X, Y, Z = plot_contour(x, y, z, resolution=500, contour_method='linear')
            with plt.style.context("classic"):
                fig, ax = plt.subplots(figsize=(13, 5))
                plt.colorbar(ax.contourf(X, Y, Z), shrink=0.7)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            plt.savefig('C:/Users/sowjanya2014_2/capstone1/deployment/Project1/static/output/plot1.png')

        else:
            print("Invalid design")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features)
    prediction = predictor(final_features[0],final_features[1],final_features[2],final_features[3],final_features[4],final_features[5],final_features[6])

    output = prediction
    full_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'plot1.png')

    return render_template('index.html', prediction_image=full_filename)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = predictor([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)