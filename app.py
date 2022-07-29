import logging
from flask import Flask, jsonify, request
import os
import numpy

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
debug = os.getenv('DEBUG')

@app.route('/curve', methods=['POST'])
def home():
    if request.method == 'POST':
        body = request.json
        logging.info(body)
        points = to_list(body['points'])
        extrapolated = points
        return jsonify({'extrapolated': extrapolated})


def to_list(bbox):
    if isinstance(bbox, numpy.ndarray):
        return bbox.tolist()
    return bbox


if __name__ == '__main__':
    start_socket_server()

