import time
import numpy as np
import pandas as pd 

def sum_trad():
    start= time.time()
    X = range(10000000)
    Y = range(10000000)
    Z = []
    for i in range(len(X)):
        Z.append(X[i] + Y[i])
    return time.time() - start 


# NumPY 
def sum_numpy():
    start= time.time()
    X = np.arange(10000000)
    Y = np.arange(10000000)
    Z = X + Y
    return time.time() - start
#print ('time sum: ', sum_trad (), 'time sum numpy :', sum_numpy() )

# arrays creation 

arr = np.array([2, 6, 5, 9 ], float)
print (arr)
arr2 = np.array([1,2,3], float)
print (arr2.tolist())
list2= list(arr2.tolist())
print (list2)

# array equal 
arr = np.array([2, 7, 5, 9 ], float)
print(np.sort(arr)) 
print(np.argsort(arr)) 
print('before randomly shuffle ', arr) 
#the whole array is shuffled.
np.random.shuffle(arr) 
print(arr)
#2d array:   it shuffles the rows, not the individual 
#  elements inside each row.
arr2d = np.arange(9).reshape(3,3)
print("Before:\n", arr2d)

np.random.shuffle(arr2d)
print("After:\n", arr2d)

arr1 = np.array([10,22], float)
arr2 = np.array([31,43,54,61], float)
arr3 = np.array([71,82,29], float)

print(np.concatenate((arr1, arr2, arr3)))

arr1 = np.array([[11, 12], [32, 42]], float)
arr2 = np.array([[54, 26], [27, 28]], float)

print(np.concatenate((arr1, arr2))) #default axis = 0 so rows stack vertically
print(np.concatenate((arr1, arr2), axis=1)) #columns get stacked (one array goes beside the other).

print (arr1.tostring())

print ( np.fromstring(arr1.tostring()))
print(arr1 + arr2)

print('Linear algreba operations')
x = np.arange(15).reshape((3,5))
print(x)
print( x.T)
print(np.dot(x.T, x)) # x^t . x 


a = np.array([1,2,3])
b = np.array([4,5,6]) 

print(' \nouter:')
print(np.outer(a, b)) # outer(a,b)[i,j]=a[i]×b[j]
print(' \n inner:') 
print(np.inner(a, b)) #dot product (skalärprodukt).
print(' \n cross: \n')
print(np.cross(a, b)) #är en ny vektor som är vinkelrät mot båda vektorerna.
#a×b=(a2​b3​−a3​b2​,a3​b1​−a1​b3​,a1​b2​−a2​b1​)
# första komponentent 2*6 - 3* 5 = -3
#andra komponentent 3*4 - 1* 6  = 6 
# tredje komponentent 1*5 - 2 *4 = -3 

matrix = np.array( [[74, 22, 10], [92,31,17], [21,22,12]], float)
print( '\nmatrix : ', matrix)

print('determinet: ', np.linalg.det(matrix), '\n')


inv_matrix = np.linalg.inv(matrix)
print ( 'inv_matrix: \n',inv_matrix,'\n\n')
 
print('*********************************')
obj = pd.Series([3,5,-2,1])
print( 'obj:\n',obj)
print('obj.values',obj.values)
print('obj index: ', obj.index)
print('obj*2\n' , obj*2)
print('obj[obj]>2',obj[obj>2] )

data = {'a' : 30, 'b' :70, 'c' : 160 , 'd': 5}
print ('data : \n', pd.Series(data))

index = {'a', 'b', 'c', 'd', 'g'}
obj2 = pd.Series(data , index = index)
print('obj2: ',obj2)

print(pd.isnull(obj2))
print(pd.notnull(obj2))

data = pd.read_csv("data_examples/data_set_lab0/Frogs_MFCCs.csv", header= None)
print(data.describe())
print(data.columns)
#print(data.dtypes)
print(data[[1,20]])
print(data[1].head(10))
print(data[1:3])

print('*********************************')

print(data[1].dtype)
print(data[1].head())

data[1] = pd.to_numeric(data[1], errors="coerce")
print(data[data[1] > 0].head(4))
print(data)