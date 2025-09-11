# Courtesy of Dominique Guillot
# This is a comment! 

# 1) Assigning variables + basic arithmetic

a = 2
b = 3
c = a + b

print(c)

# 2) Control flow: indentation matters (copy/paste: can use %paste in Ipython to preserve identation)

# If/else
if a == 2: 
	print("Yes! a=2!")
else:
	print("Sorry. a is not equal to 2")

# Loops
for i in range(10):
	print(i)

# Note: Indexing starts at 0 with Python

# Lists and tuples

a = [2,3,4]
print(a[0])
a[0] = 10
print(a)

# A tuple is constant
a = (2,3,4)
print(a)
a[0] = 10  # fails

# 3) Matrices, linear algebra

import numpy as np

# Create matrices from scratch
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

# Standard constructors
M = np.zeros((3,4))
M = np.ones((3,3))
M = np.eye(3)
M = np.random.rand(3,3)  # Random uniform [0,1] entries

print(A[0,0])  # Again, indexing starts at 0.

A+B  # Addition of matrices
A*B  # Warning "*" is entrywise (not the matrix product)

A.dot(B)  # Matrix product

# Compute eigenvalues/eigenvectors

v = np.linalg.eig(A) 

v[0]  # eigenvalues
v[1]  # eigenvectors

# Can also use 

[e, v] = np.linalg.eig(A)

# Let's verify: 
v1 = v[:,0]
v2 = v[:,1]

A.dot(v1)-e[0]*v1
A.dot(v2)-e[1]*v2

# Matrix inverse
np.linalg.inv(A)

# Solving linear systems
b = np.array([[2],[3]])
sol = np.linalg.solve(A,b)

A.dot(sol)

# 4) Plotting

x = np.linspace(-1,1,201)
y = x**2

import matplotlib.pyplot as plt

plt.plot(x,y)
plt.xlabel('Values of x')
plt.ylabel('Values of y')
plt.title("A quadratic polynomial!")
plt.show()    # Note: not required if using the Ipython notebook.

# Can also use interactive mode

plt.ion()
plt.plot(x,y, '--r')

# 5) Syntax for importing functions into environment

from numpy.linalg import eig

eig(A)

# You can also "from package import *"
# but this is generally not recommended.
