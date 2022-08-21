import logging
from flask import Flask, jsonify, request
import os
import numpy as np

from curve import fit_quadratic_drag
from magnus import fit_magnus, minimize_magnus

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
debug = os.getenv('DEBUG')


@app.route('/curve', methods=['POST'])
def home():
    if request.method == 'POST':
        body = request.json
        logging.info(body)
        points = np.array(body['points'])
        target_times = body.get('target_times')
        logging.info('points: {}'.format(points))
        extrapolated = minimize_magnus(points, target_times)
        extrapolated[:, (0, 1)] = np.round(extrapolated[:, (0, 1)], 1)
        return jsonify({'extrapolated': to_list(extrapolated)})


def to_list(bbox):
    if isinstance(bbox, np.ndarray):
        return bbox.tolist()
    return bbox
