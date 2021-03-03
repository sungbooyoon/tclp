import pickle

with open("test.pickle", "rb") as f:
    data = pickle.load(f)
print(data)
