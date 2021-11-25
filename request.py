import requests
import numpy as np

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'design':"Design3", 'app_pressure':9, 'pis_radius':6, 'pad_thickness':8, "pad_height":11,'pad_width':25,'youngs_mod':100})

print(r.json())
