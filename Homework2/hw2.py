import pandas
from decimal import Decimal

pandas.options.display.float_format = '{:.2E}'.format

#Columns and Rows for dataframe
columns = ['Start', 'learning', 'changes', 'thoroughly', 'End']
index = ['Start', 'Noun', 'Verb', 'Adverb', 'End']

#Create dataframe for calculations, initialize with 0's and Start=1
df = pandas.DataFrame(index=index, columns=columns)
df = df.fillna(0.0)
df.ix['Start', 0] = 1.0

#Transition Probabilities
trans = {
    'Start': {'Noun': 0.2, 'Verb': 0.3, 'Adverb': 0.0, 'End': 0.0},
    'Noun' : {'Noun': 0.1, 'Verb': 0.3, 'Adverb': 0.1, 'End': 0.0},
    'Verb' : {'Noun': 0.4, 'Verb': 0.1, 'Adverb': 0.4, 'End': 0.0},
    'Adverb' : {'Noun': 0.0, 'Verb': 0.0, 'Adverb': 0.0, 'End': 0.1},
    'End' : {'Noun': 0.0, 'Verb': 0.0, 'Adverb': 0.0, 'End': 0.0}
}

#Emissions Probabilites
emis = {
    'Noun' : {'learning': 0.001, 'changes': 0.003, 'thoroughly': 0.0, 'End' : 0.0},
    'Verb' : {'learning': 0.003, 'changes': 0.004, 'thoroughly': 0.0, 'End' : 0.0},
    'Adverb' : {'learning': 0.0, 'changes': 0.0, 'thoroughly': 0.002, 'End' : 0.0},
    'End' : {'learning': 0.0, 'changes': 0.0, 'thoroughly': 0.0, 'End' : 1.0}
}

#Loop through dataframe, calculating probabilities and storing max for each cell
for i in range(1, len(columns)):
    col = columns[i]
    prev_col = columns[i-1]
    #print(col)
    for j in range(1, len(index)):
        prev_row = index[j-1]
        row = index[j]
        #print(row)
        maximum = 0.0
        for k in range(len(index)):
            ix=index[k]
            #print(trans[index[j-1]][index[j]])
            viterbi = df.loc[ix, prev_col]*trans[ix][row]*emis[row][col]
            #print(viterbi)
            if viterbi > maximum:
                maximum = viterbi
        #print("max: %.12f" % maximum)
        df.ix[row, col] = maximum
        
#Print results
print(df)
print('The probability of "learning changes thoroughly" is: %.2E'%Decimal(df.loc['End']['End']))


        
     
                
    





