# Visualizing Movies

Gives you some interesting visualizations about movie viewing data.

Didn't really polish all that much. I made the wrong technical choice going for Plotly when I really did need something more flexible like D3. Oh well.

## Installation

* Install Python3 on your system
* Make a virtual env and source into it
* `pip install -r requirements.txt`
* `./src/movie_viz.py <path to input file>`

## Your input data

WONTDO: hook this up to a service like [letterboxd](https://letterboxd.com/api-beta/)

My data (obviously not included for privacy reasons) was logged in google sheets, and exported as a CSV. I logged with my partner, so the B and K columns are mine and her's. The columns I used in this viz were:
* Movie Name
* Year
* Date Watched
* K Rating
* B Rating
