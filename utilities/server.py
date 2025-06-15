from submit import *

with open("result.json", "r") as file:
    dati = json.load(file)

submit(dati)
