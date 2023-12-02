
from flask import render_template, jsonify, request, Response, Blueprint


main = Blueprint('main', __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route('/health1', methods=['POST'])
def health1():
    """
    This view is aimed to verify the healthyness of the API
    :return:
    """
    return Response('{"status":"OK"}', status=200)


@main.route('/health', methods=['GET'])
def health():
    """
    This view is aimed to verify the healthyness of the API
    :return:
    """
    return Response('{"status":"OK"}', status=200)

