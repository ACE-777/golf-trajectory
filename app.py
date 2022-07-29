import logging
from flask import Flask, jsonify, request
import os
import numpy

from curve import fit_quadratic_drag

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
debug = os.getenv('DEBUG')


@app.route('/curve', methods=['POST'])
def home():
    if request.method == 'POST':
        body = request.json
        logging.info(body)
        points = to_list(body['points'])
        target_times = body['target_times']
        logging.info('points: {}'.format(points))
        extrapolated = fit_quadratic_drag(points, target_times)
        return jsonify({'extrapolated': to_list(extrapolated)})


def to_list(bbox):
    if isinstance(bbox, numpy.ndarray):
        return bbox.tolist()
    return bbox
