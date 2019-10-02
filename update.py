import pandas as pd
import numpy as np
import re, pickle, time, json


with open("models/flatlist", "rb") as data:
    all_skills = pickle.load(data)

all_skills.append("r")
all_skills.append("infozoom")

print(len(all_skills))

with open('models/flatlist1', 'wb') as output:
    pickle.dump(all_skills, output)