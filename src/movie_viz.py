#!/usr/bin/env python3

import sys
import locale
import pdb
import time
import pickle
from pathlib import Path
import collections
import itertools
import networkx as nx

import numpy as np
import pandas as pd
import geopandas 
import imdb
import plotly.graph_objects as go
import plotly.express as px
import geopy
from geopy.extra.rate_limiter import RateLimiter

ia = imdb.IMDb()

imdb_cache_path = 'imdb_cache.p'

imdb_cache = dict()
if Path(imdb_cache_path).exists():
  imdb_cache = pickle.load(open(imdb_cache_path, 'rb'))

def imdb_search(mov_str):
  if mov_str not in imdb_cache:
    mov_id = ia.search_movie(mov_str)[0].movieID # assuming the 1st is the best match.
    full_mov = ia.get_movie(mov_id)
    # https://imdbpy.readthedocs.io/en/latest/usage/data-interface.html
    ia.update(full_mov, info=['locations', 'release dates'])
    imdb_cache[mov_str] = full_mov
  return imdb_cache[mov_str]


def get_movie_info(input_file):
  df = pd.read_csv(input_file)
  df['Date Watched'] = pd.to_datetime(df['Date Watched'])
  full_movies = df['Movie Name'].apply(lambda x: imdb_search(x))
  pickle.dump(imdb_cache, open(imdb_cache_path, 'wb'))
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
    str_loc = str_loc.replace(' USA', '')
  if ' - ' in str_loc:
    str_loc = str_loc.split(' - ')[1]
  return str_loc

def my_geocode(query):
  if query not in cache:
    cache[query] = query_geocoder_server(query)
  return cache[query]

cache_path = 'cache.p'

cache = dict()
if Path(cache_path).exists():
  cache = pickle.load(open(cache_path, 'rb'))

def query_geocoder_server(query):
  time.sleep(5)
  return geolocator.geocode(query)

geolocator = geopy.Nominatim(user_agent='movies_watched_2020')

def get_locs(df, full_movies):
  d = {'Movie Name': [], 'lat': [], 'lon': []}
  for row in df.itertuples():
    name = row._1
    print(name)
    all_locs = full_movies[row.index].get('locations')
    if all_locs is None:
      continue
    for str_loc in full_movies[row.index].get('locations'):
      str_loc = normalize_location_str(str_loc)
      loc = my_geocode(str_loc)
      if loc is None:
        print('\tSkipping ' + str_loc)
        continue
      else:
        print('Using ' + str_loc)
      d['Movie Name'].append(name)
      d['lat'].append(loc.latitude)
      d['lon'].append(loc.longitude)
      # break # For now: try again later, once we have persistant caching in place for geopy
  return pd.DataFrame(d)

def filming_locations(df, full_movies, out_f):
  loc_df = get_locs(df, full_movies) 
  pickle.dump(cache, open(cache_path, 'wb'))
  loc_df['size'] = 12
  fig = px.line_mapbox(loc_df, lat='lat', lon='lon', color='Movie Name') # , size='size')
  fig.update_layout(mapbox_style='stamen-terrain')
  out_f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
  return fig


def genres_plot(df, full_movies, out_f):
  # TODO(brycew): improve this viz
  genre_list = full_movies.apply(lambda x: x.get('genres')).explode()
  fig = px.bar(genre_list.value_counts())
  out_f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
  return fig

def budgets_violin_plot(df, full_movies, out_f):
  locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
  def grab_budget(dx):
    if 'box office' not in dx.data or 'Budget' not in dx.data['box office']:
      return np.nan
    else:
      val = dx.data['box office']['Budget'].replace(' (estimated)', '')
      mult = 1.22 if val.startswith('EUR') else 1
      return locale.atof(val.strip('$').strip('EUR')) * mult

  buds = full_movies.apply(grab_budget).dropna()
  df2 = pd.DataFrame()
  df2['budget'] = buds
  df2['name'] = df['Movie Name']
  df2 = df2.sort_values('budget')
  fig = px.violin(df2, y='budget', box=True, points='all', hover_data=df2.columns)
  out_f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
  return fig
  

def gather_cast_crew(full_movies):
  for artist_type in ['directors', 'cast']:
    artists = collections.defaultdict(int)
    for mov in full_movies:
      print(mov.data['original title'])
      if artist_type in mov.data:
        for name in mov.data[artist_type]:
          artists[name] += 1
    G = nx.Graph()
    for d_count in artists.items():
      if d_count[1] > 1:
        G.add_node(d_count[0], type=artist_type, count=d_count[1])
    for mov in full_movies:
      if artist_type in mov.data:
        all_to_add = []
        for name in mov.data[artist_type]:
          if any([node for node in G.nodes(data=True) if node[0].personID == name.personID]):
            all_to_add.append(name)
        G.add_edges_from(itertools.combinations(all_to_add, 2))
  #nx.draw_spring(G, with_labels=True)
  return G 


# TODO(brycew): at least 3 more viz:
# * graph of actors / BTS, movies are edges
#   * https://plotly.com/python/network-graphs/
# * number of women/PoC BTS and actors
# * budgets of the movies (just on a density/violin 1D plot) (movie.data['box office'][1]), strip estimated


def main(argv):
  df, full_movies = get_movie_info("../data/GoodMovies2020.csv")
  with open('plot1.html', 'w') as out_f1:
    combo_timeline(df, out_f1)
  with open('plot2.html', 'w') as out_f2:
    star_plot(df, out_f2)
  with open('plot3.html', 'w') as out_f3:
    filming_locations(df, full_movies, out_f3)


if __name__ == '__main__':
  main(sys.argv)