import numpy as np
import pandas as pd
import os
import matplotlib

os.chdir('trainingDigits')   #set workspace
goal = [0,1,2,3,4,5,6,7,8,9]    #goal (the"a")

train_data = []         #training dataset
for j in goal:
    filenum = 0         # the index of data data of a (the b)
    while True:
        tem = []
        filename = str(goal[j]) + "_" + str(filenum) + '.txt'        # get the data name ("a_b.txt")
        print("file:",filename)
        try:
            b = open(filename,'r')          # open the data file,put the data into b
        except FileNotFoundError:           # if the data file can't be found,stop getting data
            break

        c = b.read()                        #io,read b's data into c,in string
        b.close()                           #io,close b
        d = c.replace("\n",'')              #replace the "\n" from c and then get d
        size = len(d)                       #the length of d
        i = 0
        for i in range(0,size-1):
            tem.append(int(d[i]))           #change the data with string format to list
        tem.append(1)                  # add bias
        if j == 0:
            tem.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])      #if the output is 0,then add the output into tem,in "1-of-K" format
        if j == 1:
            tem.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 2:
            tem.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if j == 3:
            tem.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if j == 4:
            tem.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if j == 5:
            tem.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if j == 6:
            tem.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if j == 7:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if j == 8:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if j == 9:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        train_data.append(tem)                              #add the tem into training dataset
        filenum = filenum + 1                               #the next loop

train_data = np.array(train_data)                           #change the data with list format to array
train_data_input = train_data[:,0:-10]                      #get the input of training dataset
train_data_output = train_data[:,-10:]                      #get the output of training dataset

W = np.linalg.pinv(np.matrix(train_data_input).T*np.matrix(train_data_input))*np.matrix(train_data_input).T*np.matrix(train_data_output)


os.chdir('testDigits')
goal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
test_data = []
for j in goal:
    filenum = 0
    while True:
        tem = []
        filename = str(goal[j]) + "_" + str(filenum) + '.txt'
        print("file:",filename)
        try:
            b = open(filename,'r')
        except FileNotFoundError:
            break

        c = b.read()
        b.close()
        d = c.replace("\n",'')
        size = len(d)
        i = 0
        for i in range(0,size-1):
            tem.append(int(d[i]))
        tem.append(1)  
        if j == 0:
            tem.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 1:
            tem.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if j == 2:
            tem.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if j == 3:
            tem.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if j == 4:
            tem.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if j == 5:
            tem.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if j == 6:
            tem.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if j == 7:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if j == 8:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if j == 9:
            tem.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        test_data.append(tem)
        filenum = filenum + 1

test_data = np.array(test_data)

test_data_input = test_data[:,0:-10]                    # get the input of testing dataset
test_data_input_cal = W.T*test_data_input.T             # get the prediction

test_data_output = test_data[:,-10:]                    # get the output
shape0 = test_data_output.shape

size0 = test_data_input_cal.shape

for j in range(0,size0[1]):                             # change the prediction to 1-of-K format
    colmax = np.max(test_data_input_cal[:,j])
    #print("colmax:",colmax)
    for i in range(0, size0[0]):
        if test_data_input_cal[i,j] == colmax:
            test_data_input_cal[i,j] = 1
        else:
            test_data_input_cal[i,j] = 0

# the function(6) to get error rate
error_rate = np.sum(abs((test_data_output.T-test_data_input_cal)/2))/(shape0[0]*size0[0])
print(error_rate)





