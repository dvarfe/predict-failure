import os
from flask import Flask, render_template, redirect, send_file
from flask import request, session
from flask import url_for, jsonify

from core.system_manager import SystemManager

import pandas as pd

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.secret_key = 'your_secret_key'  # Для работы сессии

manager = SystemManager()

@app.route('/')
def main():
    # return render_template('index.html', header={})
    result = manager.find_objects()
    print(result)
    data = manager.collect_data('cpu')
    return render_template('index.html', collectors=result)

@app.route('/find_objects')
def find_objects():
    result = manager.find_objects()
    return render_template('index.html', collectors=result)

@app.route('/system_status')
def system_status():
    status = {}
    for name, collector in manager.collectors.items():
        try:
            df = collector.collect()
            if not df.empty:
                status[name] = df.iloc[0].to_dict()
        except Exception as e:
            status[name] = {"error": str(e)}
    return render_template('system_status.html', status=status)

@app.route('/feature_monitor', methods=['GET'])
def feature_monitor():
    collectors = list(manager.collectors.keys())
    selected_collector = request.args.get('collector', collectors[0] if collectors else '')
    features = []
    df = None

    if selected_collector:
        collector_obj = manager.collectors[selected_collector]
        df = collector_obj.get_history()
        features = [col for col in df.columns if col != "timestamp"]

    selected_feature = request.args.get('feature', features[0] if features else '')
    if selected_feature not in features:
        selected_feature = features[0] if features else ''
    chart_data = None

    if df is not None and selected_feature and not df.empty:
        chart_data = {
            "timestamps": df["timestamp"].tolist(),
            "values": df[selected_feature].tolist(),
        }

    return render_template(
        'feature_monitor.html',
        collectors=collectors,
        features=features,
        selected_collector=selected_collector,
        selected_feature=selected_feature,
        chart_data=chart_data
    )

if __name__ == '__main__':
    app.run(debug=True, threaded=False, host='0.0.0.0', port=11111)