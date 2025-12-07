import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional

import battery

from base.collector_base import AbstractDataCollector
from base.feature_metadata import FeatureType, FeatureMetadata


class BatteryCollector(AbstractDataCollector):
    """Коллектор данных батареи с использованием модуля battery"""

    def __init__(self, config=None):
        super().__init__(config)
        self.last_capacity = None
        self.last_time = None

    @classmethod
    def get_feature_metadata(cls) -> Dict[str, FeatureMetadata]:
        """Метаданные всех признаков батареи"""
        return {
            "timestamp": FeatureMetadata("timestamp", FeatureType.TIMESTAMP, "unix_time", "Время сбора данных"),
            "battery_id": FeatureMetadata("battery_id", FeatureType.IDENTIFIER, "", "Идентификатор батареи"),
            "device_name": FeatureMetadata("device_name", FeatureType.CATEGORICAL, "", "Имя/модель батареи"),
            "battery_present": FeatureMetadata("battery_present", FeatureType.CATEGORICAL, "bool", "Присутствует ли батарея"),
            "is_charging": FeatureMetadata("is_charging", FeatureType.CATEGORICAL, "bool", "Заряжается ли батарея"),
            "is_discharging": FeatureMetadata("is_discharging", FeatureType.CATEGORICAL, "bool", "Разряжается ли батарея"),
            "percent": FeatureMetadata("percent", FeatureType.NUMERICAL, "%", "Текущий заряд в процентах"),
            "capacity_current": FeatureMetadata("capacity_current", FeatureType.NUMERICAL, "mWh/mAh", "Текущая ёмкость"),
            "capacity_design": FeatureMetadata("capacity_design", FeatureType.NUMERICAL, "mWh/mAh", "Номинальная ёмкость"),
            "health_percent": FeatureMetadata("health_percent", FeatureType.NUMERICAL, "%", "Состояние батареи"),
            "minutes_to_empty": FeatureMetadata("minutes_to_empty", FeatureType.NUMERICAL, "min", "Время до разрядки"),
            "minutes_to_full": FeatureMetadata("minutes_to_full", FeatureType.NUMERICAL, "min", "Время до полной зарядки"),
            "charge_rate": FeatureMetadata("charge_rate", FeatureType.NUMERICAL, "%/min", "Скорость изменения заряда"),
            "status": FeatureMetadata("status", FeatureType.CATEGORICAL, "", "Статус батареи"),
        }

    def find_objects(self) -> List[str]:
        """Найти доступные батареи в системе"""
        try:
            # Проверяем наличие батареи через модуль battery
            percent = battery.percent()
            if percent is not None:
                return ["system_battery"]
            else:
                return []
        except Exception as e:
            print(f"Ошибка при поиске батарей: {e}")
            return []

    def collect(self, objects=None) -> pd.DataFrame:
        """Собрать данные по батарее"""
        timestamp = time.time()

        # Собираем данные батареи
        battery_data = self._collect_battery_data(timestamp)

        df = pd.DataFrame([battery_data])


        self.save_dataframe(df, mode='append')

        print(
            f"Собраны данные батареи: {battery_data.get('status', 'Unknown')} - {battery_data.get('percent', 'N/A')}%")
        return df

    def _collect_battery_data(self, timestamp: float) -> Dict[str, Any]:
        """Собрать все данные батареи"""
        data = {
            "timestamp": timestamp,
            "battery_id": "system_battery",
            "device_name": "System Battery",
            "battery_present": False,
            "is_charging": False,
            "is_discharging": False,
            "percent": None,
            "capacity_current": None,
            "capacity_design": None,
            "health_percent": None,
            "minutes_to_empty": None,
            "minutes_to_full": None,
            "charge_rate": None,
            "status": "No Battery",
        }

        try:
            percent = battery.percent()

            if percent is not None:
                data["battery_present"] = True
                data["percent"] = round(percent, 1)

                try:
                    is_charging = battery.is_charging()
                    is_discharging = battery.is_discharging()

                    data["is_charging"] = bool(is_charging) if is_charging is not None else False
                    data["is_discharging"] = bool(is_discharging) if is_discharging is not None else False

                    # Определяем статус
                    if data["is_charging"]:
                        data["status"] = "Charging"
                    elif data["is_discharging"]:
                        data["status"] = "Discharging"
                    elif percent >= 99:
                        data["status"] = "Full"
                    else:
                        data["status"] = "Unknown"

                except Exception as e:
                    print(f"Ошибка получения статуса зарядки: {e}")

                try:
                    current_capacity = battery.capacity()
                    design_capacity = battery.design_capacity()

                    if current_capacity is not None:
                        if isinstance(current_capacity, str):
                            current_capacity = float(current_capacity.strip().split('\n')[0])
                        data["capacity_current"] = round(float(current_capacity), 1)

                    if design_capacity is not None:
                        if isinstance(design_capacity, str):
                            design_capacity = float(design_capacity.strip().split('\n')[0])
                        data["capacity_design"] = round(float(design_capacity), 1)

                    if (data.get("capacity_current") is not None and
                        data.get("capacity_design") is not None and
                            data["capacity_design"] > 0):
                        health = (data["capacity_current"] / data["capacity_design"]) * 100
                        data["health_percent"] = round(health, 1)

                    design_capacity = None
                    try:
                        design_capacity = battery.design_capacity()
                        if isinstance(design_capacity, str):
                            design_capacity = float(design_capacity.strip().split('\n')[0])
                        data["capacity_design"] = round(float(design_capacity), 1)
                    except Exception:
                        data["capacity_design"] = None

                except Exception as e:
                    print(f"Ошибка получения ёмкости (обёртка): {e}")

                try:
                    minutes_empty = battery.minutes_to_empty()
                    minutes_full = battery.minutes_to_full()

                    if minutes_empty is not None and minutes_empty > 0:
                        data["minutes_to_empty"] = round(minutes_empty, 1)

                    if minutes_full is not None and minutes_full > 0:
                        data["minutes_to_full"] = round(minutes_full, 1)

                except Exception as e:
                    print(f"Ошибка получения времени: {e}")

                data["charge_rate"] = self._calculate_charge_rate(percent, timestamp)

        except Exception as e:
            print(f"Ошибка сбора данных батареи: {e}")

        return data

    def _calculate_charge_rate(self, current_percent: float, timestamp: float) -> Optional[float]:
        """Вычисляет скорость изменения заряда в %/мин"""
        if self.last_capacity is not None and self.last_time is not None:
            time_diff = (timestamp - self.last_time) / 60  # в минутах
            if time_diff > 0:
                percent_diff = current_percent - self.last_capacity
                charge_rate = percent_diff / time_diff
                self.last_capacity = current_percent
                self.last_time = timestamp
                return round(charge_rate, 3)

        self.last_capacity = current_percent
        self.last_time = timestamp
        return None
