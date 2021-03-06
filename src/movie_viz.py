#!/usr/bin/env python3

import sys
import locale
import pdb
import time
import pickle
import collections
import itertools
import networkx as nx
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas
import imdb
import plotly.graph_objects as go
import plotly.express as px
import geopy
from geopy.extra.rate_limiter import RateLimiter
from mako.template import Template


mov_cache_f = Path(__file__).resolve().parent.joinpath('imdb_cache.p')
imdb_cache = pickle.load(mov_cache_f.open('rb')) if mov_cache_f.exists() else dict()

ia = imdb.IMDb()
def imdb_search(movie: str):
    """Grabs the movie from the imdb cache if present, otherwise adds in to the cache"""
    if movie not in imdb_cache:
        mov_id = ia.search_movie(movie)[0].movieID  # assuming the 1st is the best match.
        full_mov = ia.get_movie(mov_id)
        # https://imdbpy.readthedocs.io/en/latest/usage/data-interface.html
        ia.update(full_mov, info=['locations', 'release dates'])
        imdb_cache[movie] = full_mov
    return imdb_cache[movie]


def get_movie_info(input_file: str):
    """Gets personal movie viewing data, and combines with IMDb data about each movie
    Schema was:
    * Movie Name
    * Year it came out
    * The last year that I watched it (B Last)
    * Date watched
    * Rating (out of 5 emoji stars)
    """
    df = pd.read_csv(input_file)
    df['Date Watched'] = pd.to_datetime(df['Date Watched'])
    full_movies = df['Movie Name'].apply(lambda x: imdb_search(x))
    pickle.dump(imdb_cache, mov_cache_f.open('wb'))
    df['index'] = df.index  # So you can index into the movie list, which can't be added to the df
    return df, full_movies


def combo_timeline(df):
    """Makes a scatter plot + some axis histrograms, X-axis = the date that we watched the movie, 
    and the Y-axis was the year it was made
    """
    df_mod = df.replace(star_vals)
    df_mod['Good'] = (df_mod['B Rating'] + df_mod['K Rating'] / 2) > 3
    # https://plotly.com/python/marginal-plots/
    fig = px.scatter(df_mod, x="Date Watched", y="Year", labels='Movie Name',
                     color='Good', hover_data=['Movie Name'],
                     marginal_x='histogram',
                     marginal_y='histogram', title='Date Watched vs Year Made')
    fig.update_layout(width=1000, height=1000) 
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# Translates emoji star ratings to numbers
star_vals = {
    "1/2": 0.5,
    "⭐": 1,
    "⭐1/2": 1.5,
    "⭐⭐": 2,
    "⭐⭐1/2": 2.5,
    "⭐⭐⭐": 3,
    "⭐⭐⭐1/2": 3.5,
    "⭐⭐⭐⭐": 4,
    "⭐⭐⭐⭐1/2": 4.5,
    "⭐⭐⭐⭐⭐": 5,
    "⭐⭐⭐⭐⭐⭐": 6,
}


def star_plot(df):
    """Tries to make a split histrogram, with my and my partner's star ratings.
    Attempted to make it an actual histrogram, but plotly didn't cooperate. 
    """
    df_mod = df.replace(star_vals)
    ratings = np.linspace(0.5, 6, 12)
    bryce_ratings = df_mod['B Rating'].value_counts().reindex(ratings).replace(np.nan, 0)
    kathy_ratings = df_mod['K Rating'].value_counts().reindex(ratings).replace(np.nan, 0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ratings, y=list(bryce_ratings), name='B', marker_color='indianred'))
    fig.add_trace(go.Bar(x=ratings, y=list(kathy_ratings), name='K', marker_color='lightsalmon'))
    fig.update_xaxes(tickvals=ratings)
    #fig.add_trace(go.Histogram(df, x='B Rating'))
    #fig.add_trace(go.Histogram(df, x='K Rating'))
    #fig.update_layout(barmode='overlay')
    #fig.update_traces(opacity=0.74) 
    # https://plotly.com/python-api-reference/generated/plotly.io.to_html.html
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def normalize_location_str(location: str) -> str:
    """Makes geocoding for IMDb locations work a little better. 
    
    The IMDb location string usually includes a lot of extra information, like ", USA",
    some specific building information, or just what scene in the movie was filmed there.
    Stripping that information out usually results in the Nominatim geocoding service
    more reliably returning the exact latitude-longitude for that location.
    """
    if location == 'USA':
        return None
    if '::' in location:
        location = location.split('::')[0]
    if ' USA' in location:
        location = location.replace(' USA', '')
    if ' - ' in location:
        location = location.split(' - ')[1]
    return location


geolocator = geopy.Nominatim(user_agent='movies_watched_2020')
def query_geocoder_server(query):
    """Enforces the Nominatim usage policy
    https://operations.osmfoundation.org/policies/nominatim/ 
    """
    time.sleep(5)
    return geolocator.geocode(query)


# Sets up a cache for nominatim queries, also necessary as part of the usage policy
geo_cache_f = Path(__file__).parent.resolve().joinpath('cache.p')
geo_cache = pickle.load(geo_cache_f.open('rb')) if geo_cache_f.exists() else dict()
def my_geocode(query):
    if query is None:
        return None
    if query not in geo_cache:
        geo_cache[query] = query_geocoder_server(query)
    return geo_cache[query]


def get_locs(df, full_movies):
    """Goes through a whole dataframe of movies and retrieves their lat-long coordinates"""
    d = {'Movie Name': [], 'lat': [], 'lon': []}
    for row in df.itertuples():
        name = row._1
        all_locs = full_movies[row.index].get('locations')
        if all_locs is None:
            continue
        for str_loc in full_movies[row.index].get('locations'):
            loc = my_geocode(normalize_location_str(str_loc))
            if loc is None:
                continue
            d['Movie Name'].append(name)
            d['lat'].append(loc.latitude)
            d['lon'].append(loc.longitude)
    return pd.DataFrame(d)


def filming_locations(df, full_movies):
    """Makes maps of all the filming locations of a movie. Pretty cool, and somehow,
    not an IMDb already? Hire me, I'll make that feature for y'all.
    """
    loc_df = get_locs(df, full_movies)
    pickle.dump(geo_cache, geo_cache_f.open('wb'))
    loc_df['size'] = 12
    all_figs = []
    for name in loc_df['Movie Name'].unique():
        temp_df = loc_df[loc_df['Movie Name'] == name]
        fig = px.scatter_mapbox(temp_df,
                                lat='lat', lon='lon', size='size',
                                color='Movie Name', color_discrete_sequence=px.colors.qualitative.G10 * 10)
        fig.update_layout(mapbox_style='stamen-terrain')
        all_figs.append( (name, fig.to_html(full_html=False, include_plotlyjs='cdn')) )
    return all_figs


def genres_plot(df, full_movies):
    """Simple bar plot of number of movies of each genre that we watched. Sorted by count."""
    genre_list = full_movies.apply(lambda x: x.get('genres')).explode()
    fig = px.bar(genre_list.value_counts())
    fig.update_layout(width=2000, height=1000) 
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def budgets_violin_plot(df, full_movies):
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
    df2['year'] = df['Year']
    df2 = df2.sort_values('budget')
    fig = px.violin(df2, y='budget', box=True, points='all', 
                    hover_data=df2.columns, title='Budgets')
    fig.update_layout(width=1000, height=1000) 
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def gather_cast_crew(full_movies):
    """ Makes a graph of actors / behind-the-scenes crew. movies are edges between them
    
    https://plotly.com/python/network-graphs/
    Didn't actually use or polish at all, it's really hard to gain any information from them.
    The networks get really dense and impossible to understand
    """
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


# Some more graph ideas that I never finished:
# * number of women/PoC behind-the-scenes crew and actors
#    * I don't think IMDb has standardized data for that (besides gender), so would have to gather by hand


def main(argv):
    if len(argv) == 2:
        movie_data_file = argv[1]
    else:
        movie_data_file = './data/GoodMovies2020.csv'
    df, full_movies = get_movie_info(movie_data_file)
    combo_timeline_graph = combo_timeline(df)
    star_plot_bar_graph = star_plot(df)
    filming_locations_graph = filming_locations(df, full_movies)
    genre_plot = genres_plot(df, full_movies)
    budgets_graph = budgets_violin_plot(df, full_movies)

    mytemplate = Template(filename="./view/webpage.html")
    with open('./view/webpage_usable.html', 'w') as f:
      f.write(mytemplate.render(combo_timeline_graph=combo_timeline_graph,
                                star_plot_bar_graph=star_plot_bar_graph,
                                filming_locations_graph=filming_locations_graph, 
                                genre_plot=genre_plot,
                                budgets_graph=budgets_graph))


if __name__ == '__main__':
    main(sys.argv)