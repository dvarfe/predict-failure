import os
import subprocess
import time
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime

import pandas as pd
import GPUtil
import pynvml

from base.collector_base import AbstractDataCollector
from base.feature_metadata import FeatureType, FeatureMetadata


class AbstractGPUProvider(ABC):
    """Абстрактный класс для работы с GPU различных производителей"""

    def __init__(self) -> None:
        self.name: str = "AbstractGPU"

    @abstractmethod
    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Обнаружить GPU данного производителя"""
        pass

    def get_unique_id(self, gpu_unique_ids: Set[str] = None) -> List[Optional[str]]:
        """Получить UUID GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_gpu_info(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получить основную информацию о GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_gpu_usage(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получить использование GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_memory_info(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получить информацию о памяти GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_temperature(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        """Получить температуру GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_power_draw(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        """Получить энергопотребление GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_fan_speed(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        """Получить скорость вентилятора GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_clock_speeds(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получить частоты GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_memory_temperature(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        """Получить температуру памяти GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_power_limits(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получить лимиты мощности GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_gpu_processes(self, gpu_unique_ids: Set[str] = None) -> List[List[Dict[str, Any]]]:
        """Получить процессы использующие GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []

    def get_encoder_decoder_utilization(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получить использование энкодера/декодера GPU для множества unique_id. Если содержит пустую строку - для всех GPU"""
        return []


class AbstractGPUDataCollector(AbstractDataCollector):
    """Базовый класс для всех GPU сборщиков"""

    @classmethod
    def get_feature_metadata(cls) -> Dict[str, FeatureMetadata]:
        """Метаданные всех признаков GPU"""
        return {
            "unique_id": FeatureMetadata("unique_id", FeatureType.IDENTIFIER, "", "Уникальный идентификатор устройства", False),
            "timestamp": FeatureMetadata("timestamp", FeatureType.TIMESTAMP, "unix_time", "Время сбора данных", False),
            "id": FeatureMetadata("id", FeatureType.NUMERICAL, "index", "Индекс GPU в системе", False),
            "name": FeatureMetadata("name", FeatureType.CATEGORICAL, "", "Название GPU", False),
            "vendor": FeatureMetadata("vendor", FeatureType.CATEGORICAL, "", "Производитель GPU", False),
            "type": FeatureMetadata("type", FeatureType.CATEGORICAL, "", "Тип GPU (discrete/integrated)", False),
            "gpu_count": FeatureMetadata("gpu_count", FeatureType.NUMERICAL, "count", "Количество GPU", True),
            "gpu_usage_percent": FeatureMetadata("gpu_usage_percent", FeatureType.NUMERICAL, "%", "Загрузка GPU", True),
            "memory_used_mb": FeatureMetadata("memory_used_mb", FeatureType.NUMERICAL, "MB", "Использованная память", True),
            "memory_total_mb": FeatureMetadata("memory_total_mb", FeatureType.NUMERICAL, "MB", "Общая память", True),
            "memory_utilization_percent": FeatureMetadata("memory_utilization_percent", FeatureType.NUMERICAL, "%", "Использование памяти", True),
            "temperature_c": FeatureMetadata("temperature_c", FeatureType.NUMERICAL, "°C", "Температура GPU", True),
            "memory_temperature": FeatureMetadata("memory_temperature", FeatureType.NUMERICAL, "°C", "Температура памяти GPU", True),
            "power_draw_w": FeatureMetadata("power_draw_w", FeatureType.NUMERICAL, "W", "Энергопотребление", True),
            "power_limit": FeatureMetadata("power_limit", FeatureType.NUMERICAL, "W", "Лимит мощности GPU", True),
            "max_power": FeatureMetadata("max_power", FeatureType.NUMERICAL, "W", "Максимальная мощность GPU", True),
            "fan_speed_percent": FeatureMetadata("fan_speed_percent", FeatureType.NUMERICAL, "%", "Скорость вентилятора", True),
            "core_clock_mhz": FeatureMetadata("core_clock_mhz", FeatureType.NUMERICAL, "MHz", "Частота ядра", True),
            "memory_clock_mhz": FeatureMetadata("memory_clock_mhz", FeatureType.NUMERICAL, "MHz", "Частота памяти", True),
            "gpu_name": FeatureMetadata("gpu_name", FeatureType.CATEGORICAL, "", "Название GPU", False),
            "driver_version": FeatureMetadata("driver_version", FeatureType.CATEGORICAL, "", "Версия драйвера", False),
            "processes": FeatureMetadata("processes", FeatureType.CATEGORICAL, "list", "Процессы использующие GPU", False),
            "utilization_encoder": FeatureMetadata("utilization_encoder", FeatureType.NUMERICAL, "%", "Использование энкодера", True),
            "utilization_decoder": FeatureMetadata("utilization_decoder", FeatureType.NUMERICAL, "%", "Использование декодера", True),
        }

    def collect(self, objects: Optional[List[str]] = None) -> pd.DataFrame:
        timestamp = time.time()
        all_gpu_data = []
        gpus_to_iterate = self.detected_gpus
        if objects:
            selected_set = set(objects)
            gpus_to_iterate = [g for g in self.detected_gpus if f"{g['name']}: {g['id']}" in selected_set]

        if gpus_to_iterate:
            # Группируем GPU по провайдерам
            providers_gpus = {}
            for gpu in gpus_to_iterate:
                provider = gpu['provider']
                if provider not in providers_gpus:
                    providers_gpus[provider] = []
                providers_gpus[provider].append(gpu)

            # Собираем данные для каждого провайдера
            for provider, gpus in providers_gpus.items():
                unique_ids = {gpu.get('unique_id', "") for gpu in gpus if gpu.get('unique_id')}
                if not unique_ids or "" in unique_ids:
                    unique_ids = set()

                gpu_info_list = provider.get_gpu_info(unique_ids)
                gpu_usage_list = provider.get_gpu_usage(unique_ids)
                memory_info_list = provider.get_memory_info(unique_ids)
                temperature_list = provider.get_temperature(unique_ids)
                power_draw_list = provider.get_power_draw(unique_ids)
                fan_speed_list = provider.get_fan_speed(unique_ids)
                clock_speeds_list = provider.get_clock_speeds(unique_ids)
                memory_temp_list = provider.get_memory_temperature(unique_ids)
                power_limits_list = provider.get_power_limits(unique_ids)
                processes_list = provider.get_gpu_processes(unique_ids)
                encoder_decoder_list = provider.get_encoder_decoder_utilization(unique_ids)

                for i, gpu in enumerate(gpus):
                    if not unique_ids:
                        result_index = i
                    else:
                        gpu_unique_id = gpu.get('unique_id', "")
                        unique_ids_list = list(unique_ids)
                        try:
                            result_index = unique_ids_list.index(gpu_unique_id)
                        except ValueError:
                            result_index = -1

                    gpu_info = gpu_info_list[result_index] if result_index >= 0 and result_index < len(gpu_info_list) else {
                    }
                    gpu_usage = gpu_usage_list[result_index] if result_index >= 0 and result_index < len(gpu_usage_list) else {
                    }
                    memory_info = memory_info_list[result_index] if result_index >= 0 and result_index < len(memory_info_list) else {
                    }
                    temperature = temperature_list[result_index] if result_index >= 0 and result_index < len(
                        temperature_list) else None
                    power_draw = power_draw_list[result_index] if result_index >= 0 and result_index < len(
                        power_draw_list) else None
                    fan_speed = fan_speed_list[result_index] if result_index >= 0 and result_index < len(
                        fan_speed_list) else None
                    clock_speeds = clock_speeds_list[result_index] if result_index >= 0 and result_index < len(clock_speeds_list) else {
                    }
                    memory_temp = memory_temp_list[result_index] if result_index >= 0 and result_index < len(
                        memory_temp_list) else None
                    power_limits = power_limits_list[result_index] if result_index >= 0 and result_index < len(power_limits_list) else {
                    }
                    processes = processes_list[result_index] if result_index >= 0 and result_index < len(processes_list) else [
                    ]
                    encoder_decoder = encoder_decoder_list[result_index] if result_index >= 0 and result_index < len(
                        encoder_decoder_list) else {}

                    gpu_data = {
                        "timestamp": timestamp,
                        "unique_id": gpu.get('unique_id', ""),
                        "id": gpu['id'],
                        "name": gpu['name'],
                        "vendor": gpu['vendor'],
                        "type": gpu['type'],
                        "gpu_count": len(self.detected_gpus),
                        "gpu_usage_percent": gpu_usage.get('usage_percent'),
                        "memory_used_mb": memory_info.get('used_mb'),
                        "memory_total_mb": memory_info.get('total_mb'),
                        "memory_utilization_percent": memory_info.get('utilization_percent'),
                        "temperature_c": temperature,
                        "memory_temperature": memory_temp,
                        "power_draw_w": power_draw,
                        "power_limit": power_limits.get("power_limit") if isinstance(power_limits, dict) else None,
                        "max_power": power_limits.get("max_power") if isinstance(power_limits, dict) else None,
                        "fan_speed_percent": fan_speed,
                        "core_clock_mhz": clock_speeds.get('core_clock') if isinstance(clock_speeds, dict) else None,
                        "memory_clock_mhz": clock_speeds.get('memory_clock') if isinstance(clock_speeds, dict) else None,
                        "gpu_name": gpu_info.get('name'),
                        "driver_version": gpu_info.get('driver_version'),
                        "processes": processes,
                        "utilization_encoder": encoder_decoder.get("utilization_encoder") if isinstance(encoder_decoder, dict) else None,
                        "utilization_decoder": encoder_decoder.get("utilization_decoder") if isinstance(encoder_decoder, dict) else None,
                    }
                    all_gpu_data.append(gpu_data)
        else:
            empty_data = {
                "timestamp": timestamp,
                "unique_id": None,
                "id": None,
                "name": None,
                "vendor": None,
                "type": None,
                "gpu_count": 0,
                "gpu_usage_percent": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "memory_utilization_percent": None,
                "temperature_c": None,
                "memory_temperature": None,
                "power_draw_w": None,
                "power_limit": None,
                "max_power": None,
                "fan_speed_percent": None,
                "core_clock_mhz": None,
                "memory_clock_mhz": None,
                "gpu_name": None,
                "driver_version": None,
                "processes": None,
                "utilization_encoder": None,
                "utilization_decoder": None,
            }
            all_gpu_data.append(empty_data)

        df = pd.DataFrame(all_gpu_data)

        write_header = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        df.to_csv(self._csv_path, mode='a', header=write_header, index=False)
        print(f"Собраны данные для {len(all_gpu_data)} GPU")
        return df

    def get_history(self):
        """Загрузить исторические данные"""
        if os.path.exists(self._csv_path):
            return pd.read_csv(self._csv_path)
        return pd.DataFrame()


class NvidiaGpuProvider(AbstractGPUProvider):
    """NVIDIA GPU провайдер"""

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "NVIDIA GPU"
        self.pynvml_initialized: bool = False
        # Сопоставление индекса реальному ID. Необходимо для сохранения идентификаторов между запусками.
        self._uuid_to_handle_cache: Dict[str, int] = {}
        self._init_pynvml()
        self._build_uuid_cache()

    def _get_gpu_indices_for_unique_ids(self, gpu_unique_ids: Set[str]) -> List[Optional[int]]:
        result = []
        for unique_id in gpu_unique_ids:
            gpu_index = self._get_gpu_index_by_uuid(unique_id)
            result.append(gpu_index)
        return result

    def _init_pynvml(self) -> None:
        """Инициализация pynvml для расширенных метрик"""
        try:
            pynvml.nvmlInit()
            self.pynvml_initialized = True
        except Exception as e:
            self.pynvml_initialized = False

    def _build_uuid_cache(self) -> None:
        self._uuid_to_handle_cache.clear()

        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            uuid_str = self._get_stable_gpu_uuid(i, gpu)
            if uuid_str:
                self._uuid_to_handle_cache[uuid_str] = i

    def _get_stable_gpu_uuid(self, index: int, gpu_obj=None) -> Optional[str]:
        if self.pynvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid_str = pynvml.nvmlDeviceGetUUID(handle)
                if uuid_str:
                    return str(uuid_str).strip()
            except Exception:
                pass

        # Если не сработало через pynvml
        if gpu_obj:
            try:
                uuid_attr = gpu_obj.uuid
                if uuid_attr:
                    return str(uuid_attr).strip()
            except Exception:
                pass

        return None

    def _get_gpu_index_by_uuid(self, uuid: str) -> Optional[int]:
        """Получение текущего индекса GPU по UUID"""
        if uuid in self._uuid_to_handle_cache:
            return self._uuid_to_handle_cache[uuid]

        self._build_uuid_cache()
        return self._uuid_to_handle_cache.get(uuid)

    def detect_gpus(self) -> List[Dict[str, Any]]:
        gpus = []
        try:
            gpu_list = GPUtil.getGPUs()
            all_unique_ids = self.get_unique_id([])
            for i, gpu in enumerate(gpu_list):
                unique_id = all_unique_ids[i] if i < len(all_unique_ids) else None
                gpus.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'vendor': 'NVIDIA',
                    'type': 'discrete',
                    'gpu_obj': gpu,
                    'unique_id': unique_id
                })
        except Exception as e:
            print(f"Ошибка обнаружения NVIDIA GPU: {e}")
        return gpus

    def get_gpu_info(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получение информации о NVIDIA GPU для множества unique_id"""
        try:
            gpus = GPUtil.getGPUs()
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                # Возвращаем информацию для всех GPU
                for gpu in gpus:
                    info = {
                        "name": gpu.name,
                        "driver_version": getattr(gpu, 'driver', 'Unknown'),
                        "vendor": "NVIDIA"
                    }
                    result.append(info)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    info = {"name": None, "driver_version": None, "vendor": "NVIDIA"}

                    if gpu_index is not None and gpu_index < len(gpus):
                        gpu = gpus[gpu_index]
                        info["name"] = gpu.name
                        info["driver_version"] = getattr(gpu, 'driver', 'Unknown')

                    result.append(info)
            return result
        except Exception as e:
            print(f"Ошибка получения информации NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_gpu_usage(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        """Получение загрузки NVIDIA GPU для множества unique_id"""
        try:
            gpus = GPUtil.getGPUs()
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                for gpu in gpus:
                    usage_info = {"usage_percent": round(gpu.load * 100, 1)}
                    result.append(usage_info)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    usage_info = {"usage_percent": None}

                    if gpu_index is not None and gpu_index < len(gpus):
                        gpu = gpus[gpu_index]
                        usage_info["usage_percent"] = round(gpu.load * 100, 1)

                    result.append(usage_info)
            return result
        except Exception as e:
            print(f"Ошибка получения загрузки NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_memory_info(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        try:
            gpus = GPUtil.getGPUs()
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                for gpu in gpus:
                    memory_info = {
                        "used_mb": int(gpu.memoryUsed),
                        "total_mb": int(gpu.memoryTotal),
                        "utilization_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1) if gpu.memoryTotal > 0 else None
                    }
                    result.append(memory_info)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    memory_info = {"used_mb": None, "total_mb": None, "utilization_percent": None}

                    if gpu_index is not None and gpu_index < len(gpus):
                        gpu = gpus[gpu_index]
                        memory_info["used_mb"] = int(gpu.memoryUsed)
                        memory_info["total_mb"] = int(gpu.memoryTotal)
                        if gpu.memoryTotal > 0:
                            memory_info["utilization_percent"] = round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1)

                    result.append(memory_info)
            return result
        except Exception as e:
            print(f"Ошибка получения памяти NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_temperature(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        try:
            gpus = GPUtil.getGPUs()
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                for gpu in gpus:
                    result.append(gpu.temperature)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    temperature = None

                    if gpu_index is not None and gpu_index < len(gpus):
                        gpu = gpus[gpu_index]
                        temperature = gpu.temperature

                    result.append(temperature)
            return result
        except Exception as e:
            print(f"Ошибка получения температуры NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_power_draw(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        if not self.pynvml_initialized:
            return [None] * (len(gpu_unique_ids) if gpu_unique_ids and "" not in gpu_unique_ids else len(GPUtil.getGPUs()) if GPUtil else 0)

        try:
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        result.append(power_mw / 1000.0)
                    except:
                        result.append(None)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    if gpu_index is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                            result.append(power_mw / 1000.0)
                        except:
                            result.append(None)
                    else:
                        result.append(None)
            return result
        except Exception as e:
            print(f"Ошибка получения мощности GPU {gpu_unique_ids}: {e}")
            return []

    def get_fan_speed(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        """Получение скорости вентилятора NVIDIA GPU"""
        if not self.pynvml_initialized:
            return [None] * (len(gpu_unique_ids) if gpu_unique_ids and "" not in gpu_unique_ids else len(GPUtil.getGPUs()) if GPUtil else 0)

        try:
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                        result.append(float(fan_speed))  # процент от максимальной скорости
                    except:
                        result.append(None)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    if gpu_index is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                            result.append(float(fan_speed))  # процент от максимальной скорости
                        except:
                            result.append(None)
                    else:
                        result.append(None)
            return result
        except Exception as e:
            print(f"Ошибка получения скорости вентилятора GPU {gpu_unique_ids}: {e}")
            return []

    def get_clock_speeds(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        if not self.pynvml_initialized:
            count = len(gpu_unique_ids) if gpu_unique_ids and "" not in gpu_unique_ids else len(
                GPUtil.getGPUs()) if GPUtil else 0
            return [{"core_clock": None, "memory_clock": None}] * count

        try:
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    clock_info = {"core_clock": None, "memory_clock": None}
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        core_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        clock_info["core_clock"] = float(core_clock)
                        memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        clock_info["memory_clock"] = float(memory_clock)
                    except:
                        pass
                    result.append(clock_info)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    clock_info = {"core_clock": None, "memory_clock": None}
                    if gpu_index is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            core_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                            clock_info["core_clock"] = float(core_clock)
                            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                            clock_info["memory_clock"] = float(memory_clock)
                        except:
                            pass
                    result.append(clock_info)
            return result
        except Exception as e:
            print(f"Ошибка получения частот GPU {gpu_unique_ids}: {e}")
            return []

    def get_memory_temperature(self, gpu_unique_ids: Set[str] = None) -> List[Optional[float]]:
        if not self.pynvml_initialized:
            return [None] * (len(gpu_unique_ids) if gpu_unique_ids and "" not in gpu_unique_ids else len(GPUtil.getGPUs()) if GPUtil else 0)

        try:
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)
                        result.append(float(mem_temp))
                    except:
                        result.append(None)
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    if gpu_index is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            mem_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)
                            result.append(float(mem_temp))
                        except:
                            result.append(None)
                    else:
                        result.append(None)
            return result
        except Exception as e:
            print(f"Ошибка получения температуры памяти NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_power_limits(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        if not self.pynvml_initialized:
            count = len(gpu_unique_ids) if gpu_unique_ids and "" not in gpu_unique_ids else len(
                GPUtil.getGPUs()) if GPUtil else 0
            return [{"power_limit": None, "max_power": None}] * count

        try:
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                        limits = {
                            "power_limit": power_limit[1] / 1000.0,
                            "max_power": power_limit[0] / 1000.0
                        }
                        result.append(limits)
                    except:
                        result.append({"power_limit": None, "max_power": None})
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    if gpu_index is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                            limits = {
                                "power_limit": power_limit[1] / 1000.0,
                                "max_power": power_limit[0] / 1000.0
                            }
                            result.append(limits)
                        except:
                            result.append({"power_limit": None, "max_power": None})
                    else:
                        result.append({"power_limit": None, "max_power": None})
            return result
        except Exception as e:
            print(f"Ошибка получения лимитов мощности NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_encoder_decoder_utilization(self, gpu_unique_ids: Set[str] = None) -> List[Dict[str, Any]]:
        if not self.pynvml_initialized:
            count = len(gpu_unique_ids) if gpu_unique_ids and "" not in gpu_unique_ids else len(
                GPUtil.getGPUs()) if GPUtil else 0
            return [{"utilization_encoder": None, "utilization_decoder": None}] * count

        try:
            result = []

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0 or "" in gpu_unique_ids:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                        decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(handle)
                        utilization = {
                            "utilization_encoder": encoder_util[0],
                            "utilization_decoder": decoder_util[0]
                        }
                        result.append(utilization)
                    except:
                        result.append({"utilization_encoder": None, "utilization_decoder": None})
            else:
                gpu_indices = self._get_gpu_indices_for_unique_ids(gpu_unique_ids)
                for gpu_index in gpu_indices:
                    if gpu_index is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                            encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                            decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(handle)
                            utilization = {
                                "utilization_encoder": encoder_util[0],
                                "utilization_decoder": decoder_util[0]
                            }
                            result.append(utilization)
                        except:
                            result.append({"utilization_encoder": None, "utilization_decoder": None})
                    else:
                        result.append({"utilization_encoder": None, "utilization_decoder": None})
            return result
        except Exception as e:
            print(f"Ошибка получения использования энкодера/декодера NVIDIA GPU {gpu_unique_ids}: {e}")
            return []

    def get_unique_id(self, gpu_unique_ids: Set[str] = None) -> List[Optional[str]]:
        """Получение стабильных UUID для NVIDIA GPU"""
        try:
            self._build_uuid_cache()

            if gpu_unique_ids is None or len(gpu_unique_ids) == 0:
                gpus = GPUtil.getGPUs()
                result = []
                for i in range(len(gpus)):
                    uuid_str = self._get_stable_gpu_uuid(i, gpus[i] if i < len(gpus) else None)
                    result.append(uuid_str)
                return result
            else:
                result = []
                for target_uuid in gpu_unique_ids:
                    gpu_index = self._get_gpu_index_by_uuid(target_uuid)
                    if gpu_index is not None:
                        result.append(target_uuid)
                    else:
                        result.append(None)
                return result

        except Exception as e:
            print(f"Ошибка получения UUID NVIDIA GPU {gpu_unique_ids}: {e}")
            return []


class IntelGpuProvider(AbstractGPUProvider):
    def __init__(self) -> None:
        super().__init__()
        self.name: str = "Intel GPU"
        self._intel_gpus: List[Dict[str, str]] = self._scan_pci_devices()

    def _scan_pci_devices(self) -> List[Dict[str, str]]:
        """Сканирование PCI устройств через /sys"""
        intel_gpus = []
        try:
            pci_path = "/sys/bus/pci/devices"
            if os.path.exists(pci_path):
                for device in os.listdir(pci_path):
                    vendor_path = os.path.join(pci_path, device, "vendor")
                    device_path = os.path.join(pci_path, device, "device")
                    class_path = os.path.join(pci_path, device, "class")

                    try:
                        # Читаем vendor ID (Intel = 0x8086)
                        with open(vendor_path, 'r') as f:
                            vendor_id = f.read().strip()

                        # Проверяем class (VGA = 0x030000)
                        with open(class_path, 'r') as f:
                            device_class = f.read().strip()

                        if vendor_id == "0x8086" and device_class.startswith("0x0300"):
                            with open(device_path, 'r') as f:
                                device_id = f.read().strip()

                            intel_gpus.append({
                                'pci_id': device,
                                'vendor_id': vendor_id,
                                'device_id': device_id,
                                'name': f"Intel GPU {device_id}"
                            })
                    except (FileNotFoundError, ValueError):
                        continue
        except Exception as e:
            print(f"Ошибка сканирования PCI: {e}")

        return intel_gpus

    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Обнаружение Intel GPU"""
        gpus = []
        for i, intel_gpu in enumerate(self._intel_gpus):
            unique_id = intel_gpu.get('pci_id') or intel_gpu.get('device_id')
            gpus.append({
                'id': i,
                'name': intel_gpu['name'],
                'vendor': 'Intel',
                'type': 'integrated',
                'pci_info': intel_gpu,
                'unique_id': unique_id
            })
        return gpus

    def get_unique_id(self, gpu_unique_ids: Set[str] = None) -> List[Optional[str]]:
        """Получение стабильных PCI идентификаторов для Intel GPU"""
        try:
            if gpu_unique_ids is None or len(gpu_unique_ids) == 0:
                result = []
                for intel_gpu in self._intel_gpus:
                    pci_id = intel_gpu.get('pci_id') or intel_gpu.get('device_id')
                    result.append(pci_id)
                return result
            else:
                result = []
                available_pci_ids = {gpu.get('pci_id') or gpu.get('device_id') for gpu in self._intel_gpus}
                for target_id in gpu_unique_ids:
                    if target_id in available_pci_ids:
                        result.append(target_id)
                    else:
                        result.append(None)
                return result
        except Exception as e:
            print(f"Ошибка получения PCI ID Intel GPU {gpu_unique_ids}: {e}")
            return []


class AmdGpuProvider(AbstractGPUProvider):
    pass


class GpuCollectorLinux(AbstractGPUDataCollector):
    """GPU коллектор для Linux"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        # Инициализируем провайдеры
        self.providers: List[AbstractGPUProvider] = [
            NvidiaGpuProvider(),
            IntelGpuProvider(),
            # AmdGpuProvider(),
        ]

        self.detected_gpus: List[Dict[str, Any]] = self._detect_available_gpus()

    def _detect_available_gpus(self) -> List[Dict[str, str]]:
        """Обнаружение доступных GPU и их провайдеров"""
        detected_gpus = []

        for provider in self.providers:
            try:
                gpus = provider.detect_gpus()
                for gpu in gpus:
                    gpu['provider'] = provider
                    detected_gpus.append(gpu)
            except Exception as e:
                print(f"Ошибка при обнаружении GPU с провайдером {provider.name}: {e}")

        print(f"Обнаружено GPU: {len(detected_gpus)}")
        for gpu in detected_gpus:
            print(f"   {gpu['vendor']} {gpu['name']} (ID: {gpu['id']})")

        return detected_gpus

    def refresh_gpu_list(self) -> List[Dict[str, str]]:
        self.detected_gpus = self._detect_available_gpus()
        return self.detected_gpus

    def find_objects(self) -> List[str]:
        return [f"{gpu['name']}: {gpu['id']}" for gpu in self.detected_gpus]

    def get_gpu_list(self) -> List[Dict[str, Union[int, str]]]:
        """Получить список всех обнаруженных GPU"""
        return [{
            'id': gpu['id'],
            'name': gpu['name'],
            'vendor': gpu['vendor'],
            'type': gpu['type'],
            'unique_id': gpu.get('unique_id')
        } for gpu in self.detected_gpus]

    def get_unique_ids(self) -> List[Optional[str]]:
        """Получить список уникальных ID всех обнаруженных GPU"""
        return [gpu.get('unique_id') for gpu in self.detected_gpus]
