import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

app = Flask(__name__)

def load_model(file='model.sav'):
	return pickle.load(open(file, 'rb'))

@app.route('/')
def HomeView():
	return render_template('homeview.html')

@app.route('/predict', methods=['POST'])
def DataEntryView():
	return render_template('predictionview.html')

@app.route('/y_predicted', methods=['GET','POST'])
def PredictedView():
	regyear = int(request.args.get('regyear'))
	powerps = float(request.args.get('powerps'))
	kms= float(request.args.get('kms'))
	regmonth = int(request.args.get('regmonth'))
	gearbox = request.args.get('geartype')
	damage = request.args.get('damage')
	model = request.args.get('model')
	brand = request.args.get('brand')
	fueltype = request.args.get('fuelType')
	vehicletype = request.args.get('vehicletype')

	new_row = {'yearOfReg':regyear, 'powerPS':powerps, 'kilometer':kms,
				'monthOfRegistration':regmonth, 'gearbox':gearbox,
				'notRepairedDamage':damage,
				'model':model, 'brand':brand, 'fuelType':fueltype,
				'vehicletype':vehicletype}

	print(new_row)

	new_df = pd.DataFrame(columns=['vehicletype','yearOfReg','gearbox',
		'powerPS','model','kilometer','monthOfRegistration','fuelType',
		'brand','notRepairedDamage'])
	new_df = new_df.append(new_row, ignore_index=True)
	labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicletype']
	mapper = {}

	for i in labels:
		mapper[i] = LabelEncoder()
		mapper[i].classes = np.load(str('classes'+i+'.npy'), allow_pickle=True)
		transform = mapper[i].fit_transform(new_df[i])
		new_df.loc[:,i+'_labels'] = pd.Series(transform, index=new_df.index)
	labeled = new_df[['yearOfReg','powerPS','kilometer','monthOfRegistration'] + [x+'_labels' for x in labels]]

	X = labeled.values.tolist()
	print('\n\n', X)
	predict = reg_model.predict(X)

	roundvalue = np.round_(predict, decimals = 2)
	print(roundvalue)

	print("Final prediction :",predict)

	return render_template('predictedview.html', predicted_value = roundvalue)

if __name__=='__main__':
	reg_model = load_model()
	app.run(debug=True)