import os
import subprocess
import time
import pandas as pd
from typing import List, Dict, Any

from base.collector_base import AbstractDataCollector
from base.feature_metadata import FeatureType, FeatureMetadata


class AbstractCPUDataCollector(AbstractDataCollector):
    """Базовый класс для всех CPU сборщиков"""

    def __init__(self, config=None):
        self.update_config(config or {})
        # Имя файла для сохранения данных — по имени класса
        self._csv_path = f"storage/data/{self.__class__.__name__}.csv"
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)

    def update_config(self, config):
        self.interval = config.get("interval", 1)  # сек между замерами

    @classmethod
    def get_feature_metadata(cls) -> Dict[str, FeatureMetadata]:
        """Метаданные всех признаков CPU"""
        return {
            "timestamp": FeatureMetadata("timestamp", FeatureType.TIMESTAMP, "unix_time", "Время сбора данных"),
            "cpu_id": FeatureMetadata("cpu_id", FeatureType.IDENTIFIER, "", "Идентификатор процессора"),
            "device_name": FeatureMetadata("device_name", FeatureType.CATEGORICAL, "", "Имя/модель процессора"),
            "cpu_usage_percent": FeatureMetadata("cpu_usage_percent", FeatureType.NUMERICAL, "%", "Загрузка процессора"),
            "cpu_idle_percent": FeatureMetadata("cpu_idle_percent", FeatureType.NUMERICAL, "%", "Простой процессора"),
            "cpu_freq_current_ghz": FeatureMetadata("cpu_freq_current_ghz", FeatureType.NUMERICAL, "GHz", "Текущая частота процессора"),
            "cpu_freq_min_ghz": FeatureMetadata("cpu_freq_min_ghz", FeatureType.NUMERICAL, "GHz", "Минимальная частота процессора"),
            "cpu_freq_max_ghz": FeatureMetadata("cpu_freq_max_ghz", FeatureType.NUMERICAL, "GHz", "Максимальная частота процессора"),
            "load_1m": FeatureMetadata("load_1m", FeatureType.NUMERICAL, "", "Средняя нагрузка за 1 минуту"),
            "load_5m": FeatureMetadata("load_5m", FeatureType.NUMERICAL, "", "Средняя нагрузка за 5 минут"),
            "load_15m": FeatureMetadata("load_15m", FeatureType.NUMERICAL, "", "Средняя нагрузка за 15 минут"),
            "load_1m_per_core": FeatureMetadata("load_1m_per_core", FeatureType.NUMERICAL, "", "Средняя нагрузка на ядро за 1 минуту"),
            "uptime_sec": FeatureMetadata("uptime_sec", FeatureType.NUMERICAL, "sec", "Время работы системы"),
            "cores": FeatureMetadata("cores", FeatureType.NUMERICAL, "count", "Количество ядер"),
            "physical_cores": FeatureMetadata("physical_cores", FeatureType.NUMERICAL, "count", "Количество физических ядер"),
            "cpu_model": FeatureMetadata("cpu_model", FeatureType.CATEGORICAL, "", "Модель процессора"),
            "cpu_vendor": FeatureMetadata("cpu_vendor", FeatureType.CATEGORICAL, "", "Производитель процессора"),
            "cache_size": FeatureMetadata("cache_size", FeatureType.NUMERICAL, "KB", "Размер кэша процессора"),
            "cpu_temp_celsius": FeatureMetadata("cpu_temp_celsius", FeatureType.NUMERICAL, "°C", "Температура процессора"),
            "total_interrupts": FeatureMetadata("total_interrupts", FeatureType.NUMERICAL, "count", "Общее количество прерываний"),
            "processes_total": FeatureMetadata("processes_total", FeatureType.NUMERICAL, "count", "Общее количество процессов"),
            "cpu_temperature_c": FeatureMetadata("cpu_temperature_c", FeatureType.NUMERICAL, "°C", "Температура процессора"),
            "context_switches": FeatureMetadata("context_switches", FeatureType.NUMERICAL, "count", "Количество переключений контекста"),
        }

    def collect(self, objects=None) -> pd.DataFrame:
        timestamp = time.time()
        load1, load5, load15 = self._get_loadavg()
        usage = self._get_cpu_usage()
        idle = self._get_cpu_idle()
        freq = self._get_cpu_freq()
        freq_min, freq_max = self._get_cpu_freq_min_max()
        uptime = self._get_uptime()
        temp = self._get_cpu_temp()
        interrupts = self._get_interrupts()
        cpu_info = self._get_cpu_info()
        cores = len(self.find_objects())
        load_1m_per_core = load1 / cores if cores else None

        processes = self._get_process_count()
        cpu_temp = self._get_cpu_temperature()
        context_switches = self._get_context_switches()

        data = {
            "timestamp": [timestamp],
            "cpu_id": ["system_cpu"],
            "device_name": [cpu_info.get('model', 'Unknown CPU')],
            "cpu_usage_percent": [usage],
            "cpu_idle_percent": [idle],
            "cpu_freq_current_ghz": [freq],
            "cpu_freq_min_ghz": [freq_min],
            "cpu_freq_max_ghz": [freq_max],
            "load_1m": [load1],
            "load_5m": [load5],
            "load_15m": [load15],
            "load_1m_per_core": [load_1m_per_core],
            "uptime_sec": [uptime],
            "cores": [cores],
            "physical_cores": [cpu_info.get('physical_cores')],
            "cpu_model": [cpu_info.get('model')],
            "cpu_vendor": [cpu_info.get('vendor')],
            "cache_size": [cpu_info.get('cache_size')],
            "cpu_temp_celsius": [temp],
            "total_interrupts": [interrupts],
            "processes_total": [processes],
            "cpu_temperature_c": [cpu_temp],
            "context_switches": [context_switches],
        }
        df = pd.DataFrame(data)
        # Сохраняем в CSV (append)
        write_header = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        df.to_csv(self._csv_path, mode='a', header=write_header, index=False)
        print("Собранные данные:", data)
        return df

    def get_history(self):
        """Загрузить исторические данные"""
        if os.path.exists(self._csv_path):
            return pd.read_csv(self._csv_path)
        return pd.DataFrame()


class CpuCollectorMacOS(AbstractCPUDataCollector):
    def find_objects(self):
        """На macOS объекты = логические CPU"""
        try:
            cores = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).decode().strip())
            return [f"cpu{i}" for i in range(cores)]
        except Exception:
            return []

    def _get_loadavg(self):
        """Load average за 1, 5, 15 минут"""
        return os.getloadavg()

    def _get_cpu_usage(self):
        """
        Используем `ps -A -o %cpu` для замера загрузки CPU (в процентах). Грубая оценка.
        """
        try:
            output = subprocess.check_output(["ps", "-A", "-o", "%cpu"]).decode().strip().split("\n")[1:]
            cpu_usages = [float(x) for x in output if x.strip()]
            return sum(cpu_usages) / os.cpu_count()
        except Exception:
            return None

    def _get_cpu_idle(self):
        try:
            output = subprocess.check_output(["sar", "-u", "1", "1"]).decode()
            for line in output.splitlines():
                if "Average" in line:
                    parts = line.split()
                    return float(parts[-1])  # %idle
        except Exception:
            return None

    def _get_cpu_freq(self):
        """Частота CPU в ГГц (номинальная и текущая)"""
        try:
            cmd = ["sysctl", "-n", "hw.cpufrequency"]
            freq_hz = int(subprocess.check_output(cmd).decode().strip())
            return freq_hz / 1e9  # ГГц
        except Exception:
            return None

    def _get_cpu_freq_min_max(self):
        # macOS не предоставляет min/max через sysctl
        return None, None

    def _get_uptime(self):
        try:
            boottime = subprocess.check_output(["sysctl", "-n", "kern.boottime"]).decode()
            sec = int(boottime.split("sec =")[1].split(",")[0].strip())
            now = int(time.time())
            return now - sec
        except Exception:
            return None

    def _get_cpu_temp(self):
        # Нет стандартного способа
        return None

    def _get_interrupts(self):
        # Нет стандартного способа
        return None

    def _get_cpu_info(self):
        try:
            model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            vendor = subprocess.check_output(["sysctl", "-n", "machdep.cpu.vendor"]).decode().strip()
            physical_cores = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
            cache_size = int(subprocess.check_output(["sysctl", "-n", "hw.l3cachesize"]).decode().strip()) // 1024
            return {
                "model": model,
                "vendor": vendor,
                "physical_cores": physical_cores,
                "cache_size": f"{cache_size} KB"
            }
        except Exception:
            return {}

    def _get_process_count(self):
        """Количество запущенных процессов"""
        try:
            output = subprocess.check_output(["ps", "-A"]).decode().strip().split("\n")
            return len(output) - 1  # Минус заголовок
        except Exception:
            return None

    def _get_cpu_temperature(self):
        """Температура CPU"""
        try:
            # Попробуем получить температуру через powermetrics
            cmd = ["sudo", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"]
            output = subprocess.check_output(cmd, timeout=2).decode()
            for line in output.split('\n'):
                if 'CPU die temperature' in line:
                    temp_str = line.split(':')[1].strip().split(' ')[0]
                    return float(temp_str)
        except Exception:
            pass
        return None

    def _get_context_switches(self):
        try:
            output = subprocess.check_output(["vm_stat"]).decode()
            for line in output.split('\n'):
                if 'CPU context switches' in line:
                    return int(line.split(':')[1].strip())
        except Exception:
            pass
        return None


class CpuCollectorLinux(AbstractCPUDataCollector):
    def find_objects(self):
        """На Linux объекты = логические CPU"""
        try:
            cores = int(subprocess.check_output(["nproc"]).decode().strip())
            return [f"cpu{i}" for i in range(cores)]
        except Exception:
            return []

    def _get_loadavg(self):
        """Load average за 1, 5, 15 минут"""
        try:
            with open('/proc/loadavg', 'r') as f:
                load_data = f.read().split()
            return float(load_data[0]), float(load_data[1]), float(load_data[2])
        except Exception:
            return 0.0, 0.0, 0.0

    def _get_cpu_usage(self):
        """Использование CPU в процентах через ps"""
        try:
            output = subprocess.check_output(["ps", "-A", "-o", "%cpu"]).decode().strip().split("\n")[1:]
            cpu_usages = [float(x) for x in output if x.strip()]
            return sum(cpu_usages) / len(self.find_objects())
        except Exception:
            return None

    def _get_cpu_idle(self):
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = line.split()
            total = sum(map(int, parts[1:]))
            idle = int(parts[4])
            if not hasattr(self, '_prev_total_idle'):
                self._prev_total_idle = total
                self._prev_idle_idle = idle
                return None
            total_diff = total - self._prev_total_idle
            idle_diff = idle - self._prev_idle_idle
            self._prev_total_idle = total
            self._prev_idle_idle = idle
            if total_diff == 0:
                return None
            idle_percent = (idle_diff / total_diff) * 100
            return idle_percent
        except Exception:
            return None

    def _get_cpu_freq(self):
        """Средняя частота CPU в ГГц"""
        try:
            # Пробуем получить текущую частоту из /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
            frequencies = [float(line.split(':')[1].strip()) for line in lines if 'cpu mhz' in line.lower()]
            if frequencies:
                return sum(frequencies) / len(frequencies) / 1000
            return None
        except Exception:
            return None

    def _get_cpu_freq_min_max(self):
        try:
            cur, minf, maxf = [], [], []
            for cpu in self.find_objects():
                num = cpu.replace('cpu', '')
                try:
                    with open(f'/sys/devices/system/cpu/cpu{num}/cpufreq/scaling_cur_freq') as f:
                        cur.append(int(f.read().strip()) / 1e6)
                    with open(f'/sys/devices/system/cpu/cpu{num}/cpufreq/scaling_min_freq') as f:
                        minf.append(int(f.read().strip()) / 1e6)
                    with open(f'/sys/devices/system/cpu/cpu{num}/cpufreq/scaling_max_freq') as f:
                        maxf.append(int(f.read().strip()) / 1e6)
                except Exception:
                    continue
            if cur and minf and maxf:
                return sum(minf)/len(minf), sum(maxf)/len(maxf)
            return None, None
        except Exception:
            return None, None

    def _get_uptime(self):
        """Время работы системы"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
            return uptime_seconds
        except Exception:
            return None

    def _get_cpu_temp(self):
        try:
            for zone in os.listdir('/sys/class/thermal'):
                if zone.startswith('thermal_zone'):
                    try:
                        with open(f'/sys/class/thermal/{zone}/type') as f:
                            if 'cpu' in f.read().lower():
                                with open(f'/sys/class/thermal/{zone}/temp') as tf:
                                    return int(tf.read().strip()) / 1000.0
                    except Exception:
                        continue
            return None
        except Exception:
            return None

    def _get_interrupts(self):
        try:
            with open('/proc/interrupts', 'r') as f:
                lines = f.readlines()
            total = 0
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 1:
                    total += int(parts[1])
            return total
        except Exception:
            return None

    def _get_cpu_info(self):
        try:
            info = {}
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
            cpu0_info = {}
            for line in lines:
                if line.strip() == '':
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    cpu0_info[key.strip()] = value.strip()
            info['model'] = cpu0_info.get('model name', 'Unknown')
            info['vendor'] = cpu0_info.get('vendor_id', 'Unknown')
            info['cache_size'] = cpu0_info.get('cache size', 'Unknown')
            # Физические ядра
            physical_cores = 0
            for line in lines:
                if line.startswith('cpu cores'):
                    physical_cores = int(line.split(':')[1].strip())
                    break
            info['physical_cores'] = physical_cores
            return info
        except Exception:
            return {}

    def _get_process_count(self):
        try:
            output = subprocess.check_output(["ps", "-A"]).decode().strip().split("\n")
            return len(output) - 1  # Минус заголовок
        except Exception:
            return None

    def _get_cpu_temperature(self):
        """Температура CPU"""
        try:
            # возможные пути к датчикам температуры
            thermal_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/hwmon/hwmon0/temp1_input',
                '/sys/class/hwmon/hwmon1/temp1_input',
            ]

            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp = int(f.read().strip())
                        return temp / 1000.0  # преобразование в градусы
        except Exception:
            pass
        return None

    def _get_context_switches(self):
        try:
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('ctxt '):
                        return int(line.split()[1])
        except Exception:
            pass
        return None
