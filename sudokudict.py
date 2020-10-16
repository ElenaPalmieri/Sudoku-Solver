import numpy as np

#Dictionary used to store already solved sudokus

#KEYS = sudoku strings as they were read
#VALUES = sudoku solutions



#Adds a new entry in the dictionary
def add(dic, key, value):
    dic[key]=value

#Searches an entry in the dictionary, similar to the key at least for the 80%
def search(dic, tofind):
    for key, value in dic.items():
        similarity = np.mean([tofind[i]==key[i] for i in range(len(key))])
        if similarity >= 0.8:
            return (key, value)
    return (False,False)

#Returns the key in matrix form
def getgrid(key):
    original = np.zeros((9,9), dtype=int)
    k=0
    for i in range(9):
        for j in range(9):
            original[i][j] = key[k]
            k+=1
    
    return original