##import numpy as nm##
import pandas as ps

column_names=['user_id', 'item_id', 'rating', 'timestamp']
dt=ps.read_csv('ml-100k/u.data', sep='\t', names=column_names)
movie_names=ps.read_csv('ml-100k/u.item', sep='\|', header=None)

movie_names=movie_names[[0, 1]]
movie_names.columns=['item_id', 'title']
dt=ps.merge(dt, movie_names, on='item_id')
movie_data=dt.pivot_table(index='user_id', values='rating', columns='title')

ratings=ps.DataFrame(dt.groupby('title').mean()['rating'])
ratings['num of ratings']=ps.DataFrame(dt.groupby('title').count()['rating'])

def predict_movies(movie_name):
    movie_ratings=movie_data[movie_name]
    similar2movie=movie_data.corrwith(movie_ratings)
    movie_simicalc=ps.DataFrame(similar2movie, columns=['correlation'])
    movie_simicalc.dropna(inplace=True)
    movie_simicalc=movie_simicalc.join(ratings['num of ratings'])
    movie_simicalc=movie_simicalc[movie_simicalc['num of ratings'] > 100]
    predict=movie_simicalc.sort_values('correlation', ascending=False)
    return list(predict.index[1:10])

print('Movie list:')
all_movies=list(movie_names['title'].sort_values())
for movie in all_movies:
    print(movie)
print()
movie_name=input("Enter movie name for a recommendation: ")
print()
if movie_name in all_movies:
    print('Similar movies:')
    predictions=predict_movies(movie_name)
    for movie in predictions:
        print(movie)
else:
    print('Movie not found.')