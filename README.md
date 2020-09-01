# Python-Numpy-Notes
#Introduction to numpy
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy


# ### Creating a numpy array

# #### numpy.array([])

# In[2]:


a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.ndim)


# ### The class of numpy is ndarray
# ##### ndarray.ndim returns the dimensions of an array

# ## np.ndim     >> gets the dimensions (number of rows)
# ## np.shape   >> gets the shape as tuple(ndim, columns)
# 

# In[3]:


a.shape


# #### np.array.size          >> returns a count of all  items in the array
# #### np.array.dtype       >> returns the data type of the array
# #### np.array.itemsize  >> returns the size of each item in memory = ndarray.dtype / ndarray.size
# #### np.array.data         >> returns the place of ndarray in memory

# In[4]:


a.size ## will produce 6 items


# In[5]:


a.itemsize ## the bits that are the size of each element = np.array.dtype(int32) / np.array.size


# In[6]:


a.dtype # data type of array


# In[7]:


a.data


# # Creating arrays

# ## There are several ways to create an ndarray in numpy:
# ### 1- np.array([sequence])
# #### We can also specify the data type of its elements
# ### 2 - np.arange(num) >> random filling
# ### There are many other ways with random and filling

# In[8]:


b = np.array(([1, 2], [7, 9]), dtype=np.int16)
b


# In[9]:


c = np.arange(20).reshape(4,5)
c


# ## np.arange(start, end, step)

# In[10]:


arange_array = np.arange(3, 20, 3)
new_range = arange_array.reshape(2, 3)
new_range


# # np.zeros((dim, rows, col))
# # np.ones((dim, rows, col))

# In[11]:


zeros_array = np.zeros((2, 3, 4)) #we can also specify data types
zeros_array


# In[12]:


ones_array = np.ones((2, 2, 3), dtype=np.int8)
ones_array


# # the ndarray...._like(cloned, fillvalue) :
# ## np.ones_like(simulated)
# ## np.zeros_like(simulated)
# ## np.full_like(simulated, fillvalue)

# In[13]:


full_array_like = np.full_like(ones_array, 5)
full_array_like


# # identity(n)
# # eye(n)

# In[14]:


id_array = np.identity(5)
id_array


# In[15]:


eyed_array = np.eye(5) 
eyed_array


# ## We can define the k for eye

# In[16]:


eye_from_two_pos = np.eye(5, k=2)
eye_from_two_pos


# # Basic Operations
# ## Operators (+, -, /, *, //, %, **)

# In[17]:


array_a = np.ones((1, 3))
array_b = np.full_like(array_a, 2)
print(array_a, array_b)


# In[18]:


array_sub = array_b - array_a
array_sub


# In[19]:


array_mult = array_a * array_b
array_mult


# In[20]:


array_add = array_a + array_b
array_add


# # Creating arrays using random

# ## np.random.rand(ciel, size=(shape))

# In[21]:


my_array = np.random.randint(500, size=(5, 100))
my_array


# # To choose randomly from existing array :
# ## np.random.choice(array_to_choose_of, size=(shape))

# In[22]:


choices = my_array[0]
my_choice = np.random.choice(choices, size=(3, 5))
my_choice


# # Statistics
# ## np.min() , np.max(), np.mean(), np.sum(), np.cumsum()

# In[23]:


my_choice_sum = my_choice.sum()
print("My choices sum is: ", my_choice_sum)
my_choice_min = my_choice.min()
print("My choices min is: ", my_choice_min)
my_choice_max = my_choice.max()
print("My choices max is: ", my_choice_max)
my_choice_mean = my_choice.mean()
print("My choices mean is: ", my_choice_mean)
my_choice_cumsum = my_choice.cumsum()
print("My choices cumsum is: ", my_choice_cumsum)


# # Indexing & Slicing

# ## np.array[row, col]

# # ** Getting the second index of all dimensions in my_array

# In[24]:


my_array[:, 1] # will produce [425, 223, 223, 55, 287]


# ## ** Getting indexes 0, 1, 2, 3, 4 from rows 0, 1, 2, 3, 4

# ## Just give tuples of indeces wanted[rows_tuple, columns_tuple]

# In[25]:


my_array[(0, 1, 2, 3, 4), (0, 1, 2, 3, 4)] # will produce [437, 223, 36, 279, 268]


# # Iterating

# ## using np.flat to iterate each single value

# In[26]:


for item in my_array.flat:
    print(item)


# ## Using np.nditer(array) to iterate each sigle value

# In[27]:


for i in np.nditer(my_array):
    print(i)


# # Reshaping

# 
# ## np.array.ravel() >> returns 1-d flattened array

# In[28]:


the_flattened_array = my_array.ravel()
the_flattened_array


# ## np.array.T >> > returns transposed array

# In[29]:


array_transposed = my_array.T
print(my_array[0, -1])
array_transposed


# # Transpose : 
# # يعني بتاخد القيم من الأول للآخر وبتحط كل قيمة في رأس العامود الاندكس صفر و لما بتخلص العواميد بتنقل على الاندكس اللي بعده و هكذ

# ## في المثال السابق أخدت القيم اللي في الرو الأول و حطيتها على رأس كل رو و بعدين نقلت على الرو التاني و حطيتها في الاندكس اللي بعده و هكذا يعني من الأخر خلت الأعمدة صفوف و الصفوف أعمدة 

# In[30]:


ar1 = np.array([[1, 2, 3], [4, 5, 6]])
ar2 = ar1.T
ar2


# ## np.array.reshape(rows, cols) >>> returns the new wanted shape

# In[31]:


array_reshaped = my_array.reshape(10, 50)
array_reshaped


# # Stacking different arrays:

# ## np.column_stack((ar1, ar2))  >>> produces 2D array

# In[32]:


array_a = np.array([[1, 2, 3],[4, 5, 6]])
array_b = np.array([[7, 8, 9], [10, 11, 12]])
new_c_array_column_stack = np.column_stack((array_a, array_b))
new_c_array_column_stack


# ## np.hstack((ar1, ar2))

# In[33]:


new_c_array_hstack = np.hstack((array_a, array_b))
new_c_array_hstack


# ## np.stack((ar1, ar2), axis = 0)

# In[34]:


new_c_array_stack_axis_zero = np.stack((array_a, array_b), axis=0)
new_c_array_stack_axis_zero


# ## np.stack((ar1, ar2), axis = 1 or axis = -2)

# In[35]:


new_c_array_stack_axis_one = np.stack((array_a, array_b), axis=1)
new_c_array_stack_axis_one


# In[38]:


new_c_array_stack_axis_minus2 = np.stack((array_a, array_b), axis=-2)
new_c_array_stack_axis_minus2


# ## np.stack((ar1, ar2), axis = 2 or axis = -1)

# In[39]:


new_c_array_stack_axis_two = np.stack((array_a, array_b), axis=2)
new_c_array_stack_axis_two


# In[40]:


new_c_array_stack_axis_minus1 = np.stack((array_a, array_b), axis=-1)
new_c_array_stack_axis_minus1


# ## np.row_stack((ar1, ar2)) & np.vstack((ar1, ar2))
# ### row_stack is vstack

# In[41]:


new_c_array_row_stack = np.row_stack((array_a, array_b))
new_c_array_row_stack


# In[42]:


new_c_array_vstack = np.vstack((array_a, array_b))
new_c_array_vstack


# # Conclusion of stacking:
# ## np.stack((a,b), axis=?) :
# ### - axis = 0 >>> just concatenates with no thing else [[a[0], [a1]], [[b[0], b[1]]].
# ### - axis = 1 or -2 >>> [[a[0], b[0]], [[a[1], b[1]]
# ### - axis = 2 or -1 >>> like np.T but in smaller range makes rows cols for similar indeces.
# 
# ## np.column_stack((a,b)) and np.hstack((a,b)) >>> merge similar indeces and produce the same ndim and shape [[a0 b0], [a1 b1]]
# 
# ## np.vstack((a,b)) and np.row_stack((a,b)) >>>> just concatenate but merge ndim [[a0],[a1],[b0],[b1]]

# # Splitting Arrays:

# In[43]:


f_array = np.arange(50)
f_array


# ## np.hsplit(array, numberofnew arrays wanted)

# In[47]:


new_f_array = np.hsplit(f_array, 5)
new_f_array


# ### we can customize each size of new wanted arrays: 
# #### eg. we want to create 4 arrays the first takes 3 values second takes an other 4 third takes to the 20th value from original and the rest to be assigned for the last array

# In[60]:


new_ff_array = np.split(f_array, (3,7,20))
new_ff_array


# In[64]:


ax = np.stack((new_f_array[0], new_f_array[-1]), axis=1)
ax


# # Indexing tricks:

# In[65]:


a = np.arange(12).reshape(3, 4)
a


# In[69]:


i = np.array([[0, 1], [1, 2]])
j = np.array([[2, 1],[3, 3]])
i, j


# In[70]:


a[i]


# In[72]:


a[i,j]


# In[74]:


a[0][2] , a[1][1], a[1][3], a[2][3]


# In[77]:


re = np.stack((i, j), axis=2)
re


# ## So, in a[i,j] :
# ### first i and j are stacked within axis 2
# ### second the stacking result is taken as indices for a

# In[79]:


a[i,2]


# # a[i,2] :
# ### first [i,2] means take i values as index 0 and take 2 as index 1 for each >> [0, 2], [1, 2], [1, 2], [2, 2]
# ### Second a[new indices]

# In[80]:


a[:, j]


# # a[:, j] :
# ### firrst [: ,j] means take each index in the original array (a) as 0 index with each index from j as index 1.
# ### resulting  the 
# ### 1D >> [[a[0, 2], a[0, 1], a[0, 3], a[0, 3]],
# ###                         2D >>   [a[1,2], a[1, 1], a[1, 3], a[1, 3]],
# ###                         3D >>   [a[2, 2], a[2, 1], a[2, 3], a[2, 3]]]

# In[ ]:

