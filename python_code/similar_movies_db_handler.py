import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

genome_scores_data = pd.read_csv('genome-scores.csv')
movies_data = pd.read_csv('movies.csv')
ratings_data = pd.read_csv('ratings.csv')

movies_df = movies_data.drop(['genres','movieId'], axis = 1)
movies_df.to_csv('movies1.csv', sep='|', header=True, index=False)


scores_pivot = genome_scores_data.pivot_table(index = ["movieId"],columns = ["tagId"],values = "relevance").reset_index()
mov_tag_df = movies_data.merge(scores_pivot, left_on='movieId', right_on='movieId', how='left')
mov_tag_df = mov_tag_df.fillna(0)
mov_tag_df = mov_tag_df.drop(['title','genres'], axis = 1)


def set_genres(genres,col):
    if genres in col.split('|'): return 1
    else: return 0


mov_genres_df = movies_data.copy()
mov_genres_df["Action"] = mov_genres_df.apply(lambda x: set_genres("Action",x['genres']), axis=1)
mov_genres_df["Adventure"] = mov_genres_df.apply(lambda x: set_genres("Adventure",x['genres']), axis=1)
mov_genres_df["Animation"] = mov_genres_df.apply(lambda x: set_genres("Animation",x['genres']), axis=1)
mov_genres_df["Children"] = mov_genres_df.apply(lambda x: set_genres("Children",x['genres']), axis=1)
mov_genres_df["Comedy"] = mov_genres_df.apply(lambda x: set_genres("Comedy",x['genres']), axis=1)
mov_genres_df["Crime"] = mov_genres_df.apply(lambda x: set_genres("Crime",x['genres']), axis=1)
mov_genres_df["Documentary"] = mov_genres_df.apply(lambda x: set_genres("Documentary",x['genres']), axis=1)
mov_genres_df["Drama"] = mov_genres_df.apply(lambda x: set_genres("Drama",x['genres']), axis=1)
mov_genres_df["Fantasy"] = mov_genres_df.apply(lambda x: set_genres("Fantasy",x['genres']), axis=1)
mov_genres_df["Film-Noir"] = mov_genres_df.apply(lambda x: set_genres("Film-Noir",x['genres']), axis=1)
mov_genres_df["Horror"] = mov_genres_df.apply(lambda x: set_genres("Horror",x['genres']), axis=1)
mov_genres_df["Musical"] = mov_genres_df.apply(lambda x: set_genres("Musical",x['genres']), axis=1)
mov_genres_df["Mystery"] = mov_genres_df.apply(lambda x: set_genres("Mystery",x['genres']), axis=1)
mov_genres_df["Romance"] = mov_genres_df.apply(lambda x: set_genres("Romance",x['genres']), axis=1)
mov_genres_df["Sci-Fi"] = mov_genres_df.apply(lambda x: set_genres("Sci-Fi",x['genres']), axis=1)
mov_genres_df["Thriller"] = mov_genres_df.apply(lambda x: set_genres("Thriller",x['genres']), axis=1)
mov_genres_df["War"] = mov_genres_df.apply(lambda x: set_genres("War",x['genres']), axis=1)
mov_genres_df["Western"] = mov_genres_df.apply(lambda x: set_genres("Western",x['genres']), axis=1)
mov_genres_df["(no genres listed)"] = mov_genres_df.apply(lambda x: set_genres("(no genres listed)",x['genres']), axis=1)
mov_genres_df.drop(['title','genres'], axis = 1, inplace=True)


def set_year(title):
    year = title.strip()[-5:-1]
    if year.isnumeric():
        return int(year)
    else:
        return 1800


def set_year_group(year):
    if (year < 1900): return 0
    elif (1900 <= year <= 1975): return 1
    elif (1976 <= year <= 1995): return 2
    elif (1996 <= year <= 2003): return 3
    elif (2004 <= year <= 2009): return 4
    elif (2010 <= year): return 5
    else: return 0


def set_rating_group(rating_counts):
    if (rating_counts <= 1): return 0
    elif (2 <= rating_counts <= 10): return 1
    elif (11 <= rating_counts <= 100): return 2
    elif (101 <= rating_counts <= 1000): return 3
    elif (1001 <= rating_counts <= 5000): return 4
    elif (5001 <= rating_counts): return 5
    else: return 0


movies = movies_data.copy()
movies['year'] = movies.apply(lambda x: set_year(x['title']), axis=1)
movies['year_group'] = movies.apply(lambda x: set_year_group(x['year']), axis=1)
movies.drop(['title','year'], axis = 1, inplace=True)


agg_movies_rat = ratings_data.groupby(['movieId']).agg({'rating': [np.size, np.mean]}).reset_index()
agg_movies_rat.columns = ['movieId','rating_counts', 'rating_mean']
agg_movies_rat['rating_group'] = agg_movies_rat.apply(lambda x: set_rating_group(x['rating_counts']), axis=1)
agg_movies_rat.drop('rating_counts', axis = 1, inplace=True)
mov_rating_df = movies.merge(agg_movies_rat, left_on='movieId', right_on='movieId', how='left')
mov_rating_df = mov_rating_df.fillna(0)
mov_rating_df.drop(['genres'], axis = 1, inplace=True)


mov_tag_df = mov_tag_df.set_index('movieId')
mov_genres_df = mov_genres_df.set_index('movieId')
mov_rating_df = mov_rating_df.set_index('movieId')


cos_tag = cosine_similarity(mov_tag_df.values)*0.5
cos_genres = cosine_similarity(mov_genres_df.values)*0.25
cos_rating = cosine_similarity(mov_rating_df.values)*0.25
cos = cos_tag+cos_genres+cos_rating


cols = mov_tag_df.index.values
inx = mov_tag_df.index
movies_sim = pd.DataFrame(cos, columns=cols, index=inx)
movies_sim.head()


def get_similar(movieId):
    df = movies_sim.loc[movies_sim.index == movieId].reset_index(). \
            melt(id_vars='movieId', var_name='sim_moveId', value_name='relevance'). \
            sort_values('relevance', axis=0, ascending=False)[1:6]
    return df


movies_similarity = pd.DataFrame(columns=['movieId','sim_moveId','relevance'])
for x in movies_sim.index.tolist():
    movies_similarity = movies_similarity.append(get_similar(x))


print(movies_similarity.head())

users_df = pd.DataFrame(ratings_data['userId'].unique(), columns=['userId'])

#create movies_df
movies_df = movies_data.drop('genres', axis = 1)
#calculate mean of ratings for each movies
agg_rating_avg = ratings_data.groupby(['movieId']).agg({'rating': np.mean}).reset_index()
agg_rating_avg.columns = ['movieId', 'rating_mean']
#merge
movies_df = movies_df.merge(agg_rating_avg, left_on='movieId', right_on='movieId', how='left')

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)"]
genres_df = pd.DataFrame(genres, columns=['genres'])

users_movies_df = ratings_data.drop('timestamp', axis = 1)

movies_genres_df = movies_data.drop('title', axis = 1)


#define a function to split genres field
def get_movie_genres(movieId):
    movie = movies_genres_df[movies_genres_df['movieId']==movieId]
    genres = movie['genres'].tolist()
    df = pd.DataFrame([b for a in [i.split('|') for i in genres] for b in a], columns=['genres'])
    df.insert(loc=0, column='movieId', value=movieId)
    return df

#create empty df
movies_genres=pd.DataFrame(columns=['movieId','genres'])
for x in movies_genres_df['movieId'].tolist():
    movies_genres=movies_genres.append(get_movie_genres(x))


#join to movies data to get genre information
user_genres_df = ratings_data.merge(movies_data, left_on='movieId', right_on='movieId', how='left')
#drop columns that will not be used
user_genres_df.drop(['movieId','rating','timestamp','title'], axis = 1, inplace=True)

def get_favorite_genre(userId):
    user = user_genres_df[user_genres_df['userId']==userId]
    genres = user['genres'].tolist()
    movie_list = [b for a in [i.split('|') for i in genres] for b in a]
    counter = Counter(movie_list)
    return counter.most_common(1)[ 0 ][ 0 ]

#create empty df
users_genres = pd.DataFrame(columns=['userId','genre'])
for x in users_df['userId'].tolist():
    users_genres = users_genres.append(pd.DataFrame([[x,get_favorite_genre(x)]], columns=['userId','genre']))


users_df.to_csv('users1.csv', sep='|', header=True, index=False)
movies_df.to_csv('movies2.csv', sep='|', header=True, index=False)
genres_df.to_csv('genres1.csv', sep='|', header=True, index=False)
users_movies_df.to_csv('users_movies1.csv', sep='|', header=True, index=False)
movies_genres.to_csv('movies_genres1.csv', sep='|', header=True, index=False)
users_genres.to_csv('users_genres1.csv', sep='|', header=True, index=False)
movies_similarity.to_csv('movies_similarity1.csv', sep='|', header=True, index=False)





