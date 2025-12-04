import os
from flask import Flask, render_template, redirect, send_file
from flask import request
from flask import url_for


from core.system_manager import SystemManager

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


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=11111)
