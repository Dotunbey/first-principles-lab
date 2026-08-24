"""
Read-only monitoring dashboard for the background self-play training run
(alphazero_selfplay.py). Deliberately does NOT touch the training process at
all - just reads the CSV log it's already writing to, so there's zero risk
to the in-progress run.

Run with: python app.py [port]
Then open http://127.0.0.1:<port>
"""

import csv
import os
import sys

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5004
LOG_FILENAME = sys.argv[2] if len(sys.argv) > 2 else 'selfplay_training_log.csv'
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', LOG_FILENAME)


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html')) as f:
        return f.read()


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


@app.route('/api/log')
def api_log():
    if not os.path.exists(LOG_PATH):
        return jsonify({'rows': []})

    rows = []
    with open(LOG_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'iteration': int(row['iteration']),
                'wins_X': int(row['wins_X']),
                'wins_O': int(row['wins_O']),
                'draws': int(row['draws']),
                'num_examples': int(row['num_examples']) if 'num_examples' in row else int(row.get('buffer_size', 0)),
                'value_loss': float(row['value_loss']),
                'policy_loss': float(row['policy_loss']),
                'value_error_vs_solver': float(row['value_error_vs_solver']),
                'elapsed_sec': float(row['elapsed_sec']),
            })
    return jsonify({'rows': rows})


if __name__ == '__main__':
    print(f"Watching log file: {LOG_PATH}")
    app.run(debug=False, port=PORT)
