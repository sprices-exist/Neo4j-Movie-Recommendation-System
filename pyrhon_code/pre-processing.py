import pandas as pd

data = pd.read_csv('ratings.csv')
df = pd.DataFrame(columns=[ 'userID', 'movieID', 'rating', 'timestamp' ])

# fields = ['userID', 'movieID', 'rating', 'timestamp'] 
new_ratings = [ ]
print("done0")
print(data.head(1))
print("done1")
m = 0
k = 100
for i, j in data.iterrows():
    if j[ 1 ] < 1001:
        m += 1
        df.loc[ m ] = [ j[ 0 ], j[ 1 ], j[ 2 ], j[ 3 ] ]
        if (k < m):
            print(m)
            k += 10000

df.to_csv(test.csv, encoding='utf-8', index=False)
