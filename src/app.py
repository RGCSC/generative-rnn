#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request

import os

#initalize our flask app
app = Flask(__name__)


@app.route('/index')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template('index.html')


