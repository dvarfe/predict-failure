import os
import pandas as pd
from flask import Flask, render_template, redirect
from flask import request
from flask import url_for

from modules.base.feature_metadata import FeatureType
from core.system_manager import SystemManager
from flask import jsonify

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.secret_key = 'your_secret_key'  # Для работы сессии

app.config['SCHEDULER_API_ENABLED'] = True

manager = SystemManager(app)


def save_settings(manager):
    for name in manager.collectors.keys():
        key = f"{name}_object"
        selected = request.form.getlist(key)

        cfg = manager.config_manager.get_collector_config(name)
        cfg['selected_objects'] = selected
        manager.config_manager.update_collector_config(name, cfg)


def collect_data(manager):
    for name in manager.collectors.keys():
        manager.collect_data(name)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'save':
            save_settings(manager)

        elif action == 'collect':
            save_settings(manager)
            collect_data(manager)
            if manager.scheduler is not None:
                manager.apply_schedule(manager.get_schedule())
            else:
                print("Scheduler не доступен. Расписание не применено.")

        return redirect(url_for('main'))

    result = manager.find_objects()
    print(result)
    schedule = manager.get_schedule()
    return render_template('index.html', collectors=result, schedule=schedule)


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


@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')


@app.route('/models', methods=['GET'])
def models():
    """Страница создания и обучения ML-моделей"""
    collectors = list(manager.collectors.keys())
    return render_template('models.html', collectors=collectors)


@app.route('/api/devices', methods=['GET'])
def api_devices():
    devices = manager.list_devices()
    return jsonify(devices)


@app.route('/api/datasets', methods=['GET'])
def api_datasets():
    device = request.args.get('device')
    datasets = manager.list_datasets(device=device)
    if device:
        return jsonify(datasets.get(device, []))
    return jsonify(datasets)


@app.route('/api/models', methods=['GET'])
def api_models():
    models_index = manager.get_models()
    device_filter = request.args.get('device', 'general')
    out = models_index.get(device_filter, [])
    return jsonify(out)


@app.route('/api/ids', methods=['GET'])
def api_ids():
    q = request.args.get('q', '')  # Запрос пользователя, по которой фильтруюется выдача
    collector = request.args.get('device')
    dataset = request.args.get('dataset')

    if not dataset:
        return jsonify([])

    df = manager.load_dataframe(dataset, device=collector)

    id_col = manager.get_id_col(collector)
    if id_col and id_col in df.columns:
        ids = df[id_col].dropna().unique().tolist()
        ids = [str(x) for x in ids]
        if q:
            ids = [x for x in ids if q.lower() in x.lower()]
    else:
        ids = []
    return jsonify(ids)


def _build_curves_from_preds(preds, id_col='id'):
    curves = {}
    if id_col not in preds.columns:
        raise ValueError(f"ID column '{id_col}' not found in predictions")
    time_cols = [c for c in preds.columns if c != id_col and c != 'time']
    preds_grouped = preds.groupby(id_col).mean()

    for idx, row in preds_grouped.iterrows() if id_col is not None else [(0, preds_grouped)]:
        survival_vals = [float(row[c]) for c in time_cols]
        curves[str(idx)] = {'time': time_cols, 'survival': survival_vals}

    return curves


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json() or {}
    device = payload.get('device') or payload.get('collector')
    dataset = payload.get('dataset')
    model_selected = payload.get('model')
    ids = payload.get('ids') or []

    if not device or not dataset or not model_selected:
        return jsonify({'error': 'device, dataset and model are required'}), 400

    model_name = None
    model_device = None
    if model_selected and '::' in model_selected:
        model_device, model_name = model_selected.split('::', 1)
    else:
        model_name = model_selected

    data = manager.load_dataframe(dataset, device=device)

    id_col = manager.get_id_col(device)

    if ids and 'all' not in ids and id_col:
        data = data[data[id_col].astype(str).isin([str(x) for x in ids])]
    if 'general' == model_device:
        preds = manager.apply_model(model_name, dataset, device_name=model_device, id_col=id_col)
    else:
        preds = manager.apply_model(model_name, dataset, id_col=id_col, ids=ids, device_name=model_device)
    print(id_col)
    curves = _build_curves_from_preds(preds, id_col=id_col)

    return jsonify({'curves': curves}), 200


@app.route('/predict/device_ids', methods=['POST'])
def predict_device_ids():
    data = request.get_json() or {}
    collector = data.get('collector')
    dataset = data.get('dataset')

    if not collector:
        return jsonify({'error': 'collector required', 'device_ids': []}), 400

    df = manager.load_dataframe(dataset, device=collector)

    id_col = manager.get_id_col(collector)
    if id_col and id_col in df.columns:
        ids = df[id_col].dropna().unique().tolist()
        ids = [str(x) for x in ids]
    else:
        ids = []

    return jsonify({'device_ids': ids}), 200


@app.route('/feature_monitor', methods=['GET'])
def feature_monitor():
    # TODO: Отображение устройств с разным id для одного девайса
    collectors = list(manager.collectors.keys())
    selected_collector = request.args.get('collector', collectors[0] if collectors else '')
    features = []
    df = None
    feature_metadata = {}

    if selected_collector:
        collector_obj = manager.collectors[selected_collector]
        df = collector_obj.get_history()

        metadata = collector_obj.get_feature_metadata()

        features = []
        for col in df.columns:
            if col in metadata and metadata[col].type == FeatureType.NUMERICAL:
                features.append(col)

        feature_metadata = {name: meta.to_dict() for name, meta in metadata.items()}

    selected_feature = request.args.get('feature', features[0] if features else '')
    if selected_feature not in features:
        selected_feature = features[0] if features else ''

    if selected_feature and selected_feature in feature_metadata:
        selected_feature_metadata = feature_metadata[selected_feature]
    chart_data = None

    if df is not None and selected_feature and not df.empty:
        device_id_column = None
        for col in df.columns:
            if col in metadata and metadata[col].type == FeatureType.IDENTIFIER:
                device_id_column = col
                break

        if device_id_column and device_id_column in df.columns:
            # Получаем объединение всех временных меток
            all_timestamps = sorted(df["timestamp"].unique())

            chart_data = []
            for device_id in df[device_id_column].unique():
                if pd.isna(device_id):
                    continue

                device_df = df[df[device_id_column] == device_id]

                if 'device_name' in device_df.columns and not pd.isna(device_df['device_name'].iloc[0]):
                    device_name = str(device_df['device_name'].iloc[0])
                else:
                    device_name = str(device_id)

                device_timestamps_values = {}
                for _, row in device_df.iterrows():
                    device_timestamps_values[row["timestamp"]] = row[selected_feature]

                values = [device_timestamps_values.get(ts, None) for ts in all_timestamps]

                chart_data.append({
                    "device_id": str(device_id),
                    "device_name": device_name,
                    "timestamps": all_timestamps,
                    "values": values,
                })

    return render_template(
        'feature_monitor.html',
        collectors=collectors,
        features=features,
        selected_collector=selected_collector,
        selected_feature=selected_feature,
        selected_feature_metadata=selected_feature_metadata,
        chart_data=chart_data
    )


@app.route('/schedule', methods=['POST'])
def schedule():
    """Формы управления запуском по расписанию
        Галочка включения расписания отвечает за то, будет ли планировщик активен
        Остальное понятно интуитивно
        Активируется при нажатии кнопки "Сохранить расписание"
    """
    try:
        schedule_enabled = request.form.get('schedule_enabled') == 'on'
        interval_value = int(request.form.get('interval_value', 5))
        interval_unit = request.form.get('interval_unit', 'minutes')
        selected_collectors = request.form.getlist('selected_collectors')

        schedule_config = {
            'enabled': schedule_enabled,
            'interval_value': interval_value,
            'interval_unit': interval_unit,
            'selected_collectors': selected_collectors
        }

        manager.set_schedule(
            enabled=schedule_enabled,
            interval_value=interval_value,
            interval_unit=interval_unit,
            selected_collectors=selected_collectors)

        print(f"Настройки расписания сохранены: {schedule_config}")
        return redirect(url_for('main'))

    except Exception as e:
        print(f"Ошибка при сохранении настроек расписания: {e}")
        # TODO: более красивый хэндлер ошибок
        raise e
        # return redirect(url_for('main'))


@app.route('/stop_schedule', methods=['POST'])
def stop_schedule():
    manager.set_schedule(enabled=False)
    if manager.scheduler is not None:
        manager.apply_schedule(manager.get_schedule())
    return redirect(url_for('main'))


@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """API для обучения новых моделей"""
    try:
        payload = request.get_json() or {}
        model_type = payload.get('model_type')
        model_name = payload.get('model_name')
        device = payload.get('device')
        dataset = payload.get('dataset')
        parameters = payload.get('parameters', {})

        if not all([model_type, model_name, device, dataset]):
            return jsonify({'error': 'Не все обязательные параметры указаны'}), 400

        model = manager.fit_model(model_type=model_type, data=dataset, params=parameters, device_name=device)

        if model is None:
            return jsonify({'error': 'Не удалось обучить модель. Проверьте параметры и данные.'}), 400

        manager.save_model(model, device, model_name)

        return jsonify({
            'success': True,
            'message': f'Модель {model_name} успешно обучена и сохранена',
            'model_name': model_name,
            'model_type': model_type,
            'device': device
        })
    except Exception as e:
        return jsonify({'error': f'Ошибка при обучении модели: {str(e)}'}), 500


@app.route('/api/available_models', methods=['GET'])
def api_available_models():
    """API для получения списка доступных типов моделей"""
    try:
        models = manager.list_models_to_fit()
        print(models)
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': f'Ошибка получения списка моделей: {str(e)}'}), 500


@app.route('/api/model_parameters/<model_type>', methods=['GET'])
def api_model_parameters(model_type):
    """API для получения параметров конкретного типа модели"""
    try:
        parameters = manager.get_model_parameters(model_type)
        return jsonify(parameters)
    except Exception as e:
        return jsonify({'error': f'Ошибка получения параметров модели: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False, threaded=False, host='0.0.0.0', port=11111)
