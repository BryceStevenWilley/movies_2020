#!/usr/bin/env python3

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import imdb
import geopandas 
import pdb
import geopy

ia = imdb.IMDb()

def get_movie_info(input_file):
  df = pd.read_csv('data/GoodMovies2020.csv')
  df['Date Watched'] = pd.to_datetime(df['Date Watched'])
  all_movie_ids = df['Movie Name'].apply(lambda x: ia.search_movie(x)[0].movieID) # assuming the 1st is the best match.
  full_movies = [ia.get_movie(mID) for mID in all_movie_ids]
  # https://imdbpy.readthedocs.io/en/latest/usage/data-interface.html
  [ia.update(movie_obj, info=['locations', 'release dates']) for movie_obj in full_movies]
  df['index'] = df.index # So you can index into the movie list, which can't be added to the df
  return df, full_movies

# some cool sns stuff, not necessary
# sns.jointplot(x='Date Watched', y='Year', data=df, kind='scatter', color='#4cb391'); plt.show();

def combo_timeline(df, output_f):
  # https://plotly.com/python/marginal-plots/
  fig = px.scatter(df, x="Date Watched", y="Year", marginal_x='histogram', marginal_y='histogram', title='Test')
  output_f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
  return fig


star_vals = {
"1/2": 0.5,
"⭐": 1,
"⭐1/2": 1.5,
"⭐⭐": 2,
"⭐⭐1/2":2.5,
"⭐⭐⭐": 3,
"⭐⭐⭐1/2": 3.5,
"⭐⭐⭐⭐": 4,
"⭐⭐⭐⭐1/2": 4.5,
"⭐⭐⭐⭐⭐": 5,
"⭐⭐⭐⭐⭐⭐":6,
}

def star_plot(df, output_f):
  df_mod = df.replace(star_vals)
  ratings = np.linspace(0.5, 6, 12)
  fig = go.Figure()
  fig.add_trace(go.Bar(x=ratings, y=list(df['Bryce Rating'].value_counts().reindex(ratings).replace(np.nan, 0)), name='Bryce', marker_color='indianred'))
  fig.add_trace(go.Bar(x=ratings, y=list(df['Kathy Rating'].value_counts().reindex(ratings).replace(np.nan, 0)), name='Kathy', marker_color='lightsalmon'))
  fig.update_xaxes(tickvals=ratings)
  #fig.add_trace(go.Histogram(df, x='Bryce Rating'))
  #fig.add_trace(go.Histogram(df, x='Kathy Rating'))
  #fig.update_layout(barmode='overlay')
  #fig.update_traces(opacity=0.74) 
  # https://plotly.com/python-api-reference/generated/plotly.io.to_html.html
  output_f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
  return fig

def normalize_location_str(str_loc):
  if '::' in str_loc:
    str_loc = str_loc.split('::')[0]
  if ' USA' in str_loc:
    str_loc = str_loc.removesuffix(' USA')
  if ' - ' in str_loc:
    str_loc = str_loc.split(' - ')[1]
  return str_loc

called_set = set()


def get_locs(df_x, full_movies):
  d = {'Movie Name': [], 'lat': [], 'lon': []}
  if 'Movie Name' not in df_x.columns:
    # ??? Why does this happen?
    pdb.set_trace()
    return pd.DataFrame(d)
  print(df_x)
  name = df_x.iloc[0]['Movie Name']
  if name in called_set:
    pdb.set_trace()
  else:
    called_set.add(name)
  locator = geopy.Nominatim(user_agent='movies_2020')
  for str_loc in full_movies[df_x.iloc[0]['index']].get('locations'):
    str_loc = normalize_location_str(str_loc)
    loc = locator.geocode(str_loc)
    if loc is None:
      print('\tSkipping ' + str_loc)
      continue
    else:
      print('Using ' + str_loc)
    d['Movie Name'].append(name)
    d['lat'].append(loc.latitude)
    d['lon'].append(loc.longitude)
  return pd.DataFrame(d)

def filming_locations(df, full_movies, out_f):
  loc_df = df.groupby('Movie Name', group_keys=False).apply(lambda df_x: get_locs(df_x, full_movies)).reset_index(drop=True)

# TODO(brycew): at least 3 more viz:
# * map of filming locations (movie.data['locations'])
# * graph of actors / BTS, movies are edges
#   * https://plotly.com/python/network-graphs/
# * genre-mashing count: (movie.data['genres'])
# * number of women/PoC BTS and actors
# * budgets of the movies (just on a density/violin 1D plot) (movie.data['box office'][1]), strip estimated


def main(argv):
  df, full_movies = get_movie_info("data/GoodMovies2020.csv")
  with open('plot1.html', 'w') as out_f1:
    combo_timeline(df, out_f1)
  with open('plot2.html', 'w') as out_f2:
    star_plot(df, out_f2)


if __name__ == '__main__':
  main(sys.argv)