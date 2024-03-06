import numpy as np
import pandas as pd

df = pd.read_csv('Documents\\IML\\task0_sl19d1\\test.csv')
#print(df.to_string())

#print(df.iloc[:, 1:11].mean(axis=1))
answer = {'Id': df.iloc[:, 0], 'y': df.iloc[:, 1:11].mean(axis=1)}
dfAnswer = pd.DataFrame(answer)
dfAnswer.to_csv('Documents\\IML\\task0_sl19d1\\answer.csv', index=False)