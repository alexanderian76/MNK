from email.quoprimime import header_check
from multiprocessing.connection import wait
import numpy as np
import matplotlib.pyplot as plot
import csv
import pandas as pd
import seaborn as sn

#csv.reader
#plot.plot([1,2,3], [1,2,3])
#plot.show()
Y = []
fullY = []
fullA = []
A = []
dates = []
i = 0
with open('cpi.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['Y'])
        if i > 22:
            Y.append(float(row['Y']))
            A.append([float(row['X1']), float(row['X2']), float(row['X3']), float(row['X4']), float(row['X5'])])
        i += 1

with open('cpi.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['Y'])
        fullY.append(float(row['Y']))
        fullA.append([float(row['X1']), float(row['X2']), float(row['X3']), float(row['X4']), float(row['X5'])])

with open('dates.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        dates.append(row[0])

print(dates)

#print(Y)
#print(A)

print(len(A))
print(len(Y))
Y1 = []
result = np.linalg.lstsq(A, Y, rcond=-1)
for item in fullA:
    Y1.append(float(item[0]*result[0][0] + item[1]*result[0][1] + item[2]*result[0][2] + item[3]*result[0][3] + item[4]*result[0][4]))
print(result[0])

dateValues = []
for num in range(0, 96):
    dateValues.append(num)

fullY.reverse()
Y1.reverse()
dates.reverse()

dates = pd.to_datetime(dates).strftime("%m/%d/%Y")
DF = pd.DataFrame()
DF['value'] = fullY
DF = DF.set_index(dates)


plot.subplot(1,2,1)
#plot.bar(dates, dateValues)
plot.plot(DF, label='Реальные значения CPI')
plot.plot(Y1, label='Прогнозные значения CPI')
plot.gcf().autofmt_xdate()
plot.xticks(np.arange(0, 96, step=10))
plot.legend()



plot.subplot(1,2,2)
DATA = pd.read_csv("cpi.csv")



# Numeric columns of the dataset
numeric_col = ['X1','X2','X3','X4','X5']

# Correlation Matrix formation
corr_matrix = DATA.loc[:,numeric_col].corr()
print(corr_matrix.to_numpy().transpose() * corr_matrix)

print(np.linalg.det(corr_matrix.to_numpy().transpose() * corr_matrix))
#Using heatmap to visualize the correlation matrix
sn.heatmap(corr_matrix, annot=True)

#plot.show()
