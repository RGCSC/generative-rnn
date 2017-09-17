from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# @app.route('/index')
# def form():
#     return render_template('index.html')

# @app.route('/index', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         asd = request.json
#         print(asd)
#         session['formdata'] = asd
#         if 'formdata' in session:
#             return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
#     return render_template("index.html")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/_submit_problem', methods=["GET", "POST"])
def add_numbers():
    """Add two numbers server side, ridiculous but well..."""
    a = request.args.get('userInput', 0, type=str)

    return jsonify(result=a)


@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
