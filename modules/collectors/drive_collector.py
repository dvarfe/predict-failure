from abc import abstractmethod
import os
import re
import time
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from ..base.collector_base import AbstractDataCollector
from ..base.feature_metadata import FeatureType, FeatureMetadata

from pySMART import Device as PySMART_Device, DeviceList as PySMART_DeviceList  # type: ignore
from pySMART.interface import AtaAttributes, NvmeAttributes, SCSIAttributes


class AbstractDriveDataCollector(AbstractDataCollector):
    """Базовый класс для всех сборщиков данных SMART дисков"""

    def __init__(self, config=None):
        super().__init__(config)
        # Инициализируется в подклассах
        self.detected_drives: List[Dict[str, Any]] = []

    @classmethod
    def get_feature_metadata(cls) -> Dict[str, FeatureMetadata]:
        """Метаданные всех признаков дисков"""
        return {
            "timestamp": FeatureMetadata("timestamp", FeatureType.TIMESTAMP, "unix_time", "Время сбора данных"),
            "device": FeatureMetadata("device", FeatureType.CATEGORICAL, "", "Устройство диска"),
            "device_name": FeatureMetadata("device_name", FeatureType.CATEGORICAL, "", "Имя/модель диска"),
            "serial_number": FeatureMetadata("serial_number", FeatureType.IDENTIFIER, "", "Серийный номер диска"),
            "model_name": FeatureMetadata("model_name", FeatureType.CATEGORICAL, "", "Модель диска"),
            "firmware_version": FeatureMetadata("firmware_version", FeatureType.CATEGORICAL, "", "Версия прошивки"),
            "capacity_bytes": FeatureMetadata("capacity_bytes", FeatureType.NUMERICAL, "bytes", "Объем диска"),

            # SMART статус
            "smart_status": FeatureMetadata("smart_status", FeatureType.CATEGORICAL, "", "Общий SMART статус"),

            # Общие атрибуты
            "temperature": FeatureMetadata("temperature", FeatureType.NUMERICAL, "°C", "Температура диска"),
            "power_on_hours": FeatureMetadata("power_on_hours", FeatureType.NUMERICAL, "hours", "Часы работы"),
            "power_cycles": FeatureMetadata("power_cycles", FeatureType.NUMERICAL, "count", "Циклы питания"),

            # NVMe специфичные атрибуты
            "nvme_percentage_used": FeatureMetadata("nvme_percentage_used", FeatureType.NUMERICAL, "%", "Процент использования NVMe"),
            "nvme_data_units_written": FeatureMetadata("nvme_data_units_written", FeatureType.NUMERICAL, "units", "Записанных блоков NVMe"),
            "nvme_data_units_read": FeatureMetadata("nvme_data_units_read", FeatureType.NUMERICAL, "units", "Прочитанных блоков NVMe"),
            "nvme_integrity_errors": FeatureMetadata("nvme_integrity_errors", FeatureType.NUMERICAL, "count", "Ошибки целостности NVMe"),
            "nvme_unsafe_shutdowns": FeatureMetadata("nvme_unsafe_shutdowns", FeatureType.NUMERICAL, "count", "Небезопасные выключения NVMe"),
            "nvme_available_spare": FeatureMetadata("nvme_available_spare", FeatureType.NUMERICAL, "%", "Доступный резерв NVMe"),
            "nvme_critical_warning": FeatureMetadata("nvme_critical_warning", FeatureType.NUMERICAL, "flags", "Критические предупреждения NVMe"),

            # ATA специфичные атрибуты
            "ata_temperature_celsius_raw": FeatureMetadata("ata_temperature_celsius_raw", FeatureType.NUMERICAL, "", "ATA температура (сырое значение)"),
            "ata_temperature_celsius_value": FeatureMetadata("ata_temperature_celsius_value", FeatureType.NUMERICAL, "°C", "ATA температура"),
            "ata_power_on_hours_raw": FeatureMetadata("ata_power_on_hours_raw", FeatureType.NUMERICAL, "", "ATA часы работы (сырое значение)"),
            "ata_power_on_hours_value": FeatureMetadata("ata_power_on_hours_value", FeatureType.NUMERICAL, "hours", "ATA часы работы"),
            "ata_power_cycle_count_raw": FeatureMetadata("ata_power_cycle_count_raw", FeatureType.NUMERICAL, "", "ATA циклы питания (сырое значение)"),
            "ata_power_cycle_count_value": FeatureMetadata("ata_power_cycle_count_value", FeatureType.NUMERICAL, "count", "ATA циклы питания"),

            # Ошибки и производительность
            "read_error_rate": FeatureMetadata("read_error_rate", FeatureType.NUMERICAL, "rate", "Частота ошибок чтения"),
            "seek_error_rate": FeatureMetadata("seek_error_rate", FeatureType.NUMERICAL, "rate", "Частота ошибок позиционирования"),
            "spin_up_time": FeatureMetadata("spin_up_time", FeatureType.NUMERICAL, "ms", "Время раскрутки"),
            "start_stop_count": FeatureMetadata("start_stop_count", FeatureType.NUMERICAL, "count", "Циклы старт-стоп"),

            # Для SSD
            "wear_leveling_count": FeatureMetadata("wear_leveling_count", FeatureType.NUMERICAL, "count", "Износ выравнивания"),
            "program_fail_count": FeatureMetadata("program_fail_count", FeatureType.NUMERICAL, "count", "Ошибки программирования"),
            "erase_fail_count": FeatureMetadata("erase_fail_count", FeatureType.NUMERICAL, "count", "Ошибки стирания"),
            "ssd_life_left": FeatureMetadata("ssd_life_left", FeatureType.NUMERICAL, "%", "Остаток жизни SSD"),

            # Дополнительные метрики
            "throughput_performance": FeatureMetadata("throughput_performance", FeatureType.NUMERICAL, "rate", "Производительность"),
            "seek_time_performance": FeatureMetadata("seek_time_performance", FeatureType.NUMERICAL, "rate", "Производительность позиционирования"),
            "spin_retry_count": FeatureMetadata("spin_retry_count", FeatureType.NUMERICAL, "count", "Повторы раскрутки"),
            "calibration_retry_count": FeatureMetadata("calibration_retry_count", FeatureType.NUMERICAL, "count", "Повторы калибровки"),

            # Статистика использования
            "total_lbas_written": FeatureMetadata("total_lbas_written", FeatureType.NUMERICAL, "count", "Всего записано LBA"),
            "total_lbas_read": FeatureMetadata("total_lbas_read", FeatureType.NUMERICAL, "count", "Всего прочитано LBA"),

            # Дополнительная информация
            "drive_type": FeatureMetadata("drive_type", FeatureType.CATEGORICAL, "", "Тип диска"),
            "interface": FeatureMetadata("interface", FeatureType.CATEGORICAL, "", "Интерфейс диска"),
        }

    def find_objects(self) -> List[str]:
        """Найти доступные диски для мониторинга"""
        self.detected_drives = self._detect_drives()
        try:
            return [device['device'] for device in self.detected_drives]
        except Exception as e:
            return []

    def collect(self, objects: Optional[List[str]] = None) -> pd.DataFrame:
        """Собрать SMART данные всех дисков"""
        timestamp = time.time()
        all_drive_data = []

        drives_to_iterate = self.detected_drives
        if objects:
            drives_to_iterate = [d for d in self.detected_drives if d.get('device') in objects]
        if not drives_to_iterate:
            empty_data = self._get_empty_drive_data(timestamp)
            all_drive_data.append(empty_data)
        else:
            for drive in drives_to_iterate:
                drive_data = self._collect_drive_smart_data(drive, timestamp)
                all_drive_data.append(drive_data)

        all_keys = set()
        for row in all_drive_data:
            all_keys.update(row.keys())

        for row in all_drive_data:
            for k in all_keys:
                if k not in row:
                    row[k] = None

        df = pd.DataFrame(all_drive_data)

        self.save_dataframe(df, mode='append')

        return df

    def _collect_drive_smart_data(self, drive: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Сбор SMART данных для одного диска"""
        device_name = drive['device']
        device_obj = drive.get('_device_obj')

        # Базовые данные диска
        data = {
            "timestamp": timestamp,
            "device": device_name,
            "device_name": drive.get('model_name', f"Drive {device_name}"),
            "serial_number": drive.get('serial_number'),
            "model_name": drive.get('model_name'),
            "firmware_version": drive.get('firmware_version'),
            "capacity_bytes": drive.get('capacity_bytes'),
            "drive_type": drive.get('drive_type'),
            "interface": drive.get('interface'),
        }

        try:
            smart_data = self._get_smart_data_from_device_obj(device_obj)
            data.update(smart_data)
        except Exception as e:
            print(f"Ошибка при получении SMART данных для устройства {device_name}: {e}")
            smart_attributes = self._get_empty_smart_attributes()
            data.update(smart_attributes)

        return data

    def _get_empty_smart_attributes(self) -> Dict[str, Any]:
        """Возвращает словарь с пустыми SMART атрибутами"""
        return {
            "smart_status": None,
        }

    def _get_empty_drive_data(self, timestamp: float) -> Dict[str, Any]:
        """Возвращает пустую строку данных когда дисков нет"""
        empty_data = {
            "timestamp": timestamp,
            "device": None,
            "serial_number": None,
            "model_name": None,
            "firmware_version": None,
            "capacity_bytes": None,
            "drive_type": None,
            "interface": None,
        }
        empty_data.update(self._get_empty_smart_attributes())
        return empty_data

    def refresh_drives_list(self) -> List[Dict[str, Any]]:
        self.detected_drives = self._detect_drives()
        return self.detected_drives

    def get_drive_list(self) -> List[Dict[str, Union[str, int, float]]]:
        return [{
            'device': drive['device'],
            'model_name': drive.get('model_name'),
            'serial_number': drive.get('serial_number'),
            'capacity_gb': drive.get('capacity_gb'),
            'drive_type': drive.get('drive_type'),
            'interface': drive.get('interface')
        } for drive in self.detected_drives]

    @abstractmethod
    def _detect_drives(self) -> List[Dict[str, Any]]:
        """Обнаружение дисков"""
        pass

    @abstractmethod
    def _get_smart_data_from_device_obj(self, device_obj) -> Dict[str, Any]:
        """Получение SMART данных из объекта устройства"""
        pass


class DriveCollectorLinux(AbstractDriveDataCollector):
    """Коллектор SMART данных для дисков в Linux"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.detected_drives = self._detect_drives()

    def _detect_drives(self) -> List[Dict[str, Any]]:
        """Обнаружение всех доступных дисков через pySMART DeviceList"""
        drives = []
        try:
            device_list = PySMART_DeviceList()

            for device in device_list.devices:
                try:
                    drive_info = self._extract_drive_info(device)
                    if drive_info:
                        drives.append(drive_info)
                except Exception as e:
                    print(f"Ошибка при получении информации о диске {device.name}: {e}")
                    continue

        except Exception as e:
            print(f"Ошибка при обнаружении дисков: {e}")

        print(f"Обнаружено дисков: {len(drives)}")
        for drive in drives:
            model = drive.get('model_name') or 'Unknown'
            cap_gb = drive.get('capacity_gb', 0)
            print(f"  {drive['device']}: {model} ({cap_gb:.1f} GB)")

        return drives

    def _extract_drive_info(self, device: PySMART_Device) -> Optional[Dict[str, Any]]:
        info = {}
        info['device'] = device.name
        info['model_name'] = device.model
        info['serial_number'] = device.serial
        info['firmware_version'] = device.firmware

        info['smart_capable'] = device.smart_capable

        if not device.smart_capable or not device.smart_enabled:
            print(f"  Диск {device.name} не поддерживает SMART или SMART отключен")
            return None

        capacity_bytes = device._capacity
        if capacity_bytes is not None:
            info['capacity_bytes'] = capacity_bytes
            info['capacity_gb'] = capacity_bytes / (1024**3)
        else:
            info['capacity_bytes'] = None
            info['capacity_gb'] = 0.0

        interface = device.interface
        if interface:
            info['interface'] = interface.upper()

            if interface.lower() in ['nvme']:
                info['drive_type'] = 'NVMe'
            elif device.is_ssd:
                info['drive_type'] = 'SSD'
            elif device.rotation_rate and device.rotation_rate > 0:
                info['drive_type'] = 'HDD'
            else:
                info['drive_type'] = 'Unknown'
        else:
            info['interface'] = 'Unknown'
            info['drive_type'] = 'Unknown'

        info['_device_obj'] = device

        return info

    def _collect_drive_smart_data(self, drive: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Сбор SMART данных для одного диска"""
        device_name = drive['device']
        device_obj = drive.get('_device_obj')

        # Базовые данные диска
        data = {
            "timestamp": timestamp,
            "device": device_name,
            "serial_number": drive.get('serial_number'),
            "model_name": drive.get('model_name'),
            "firmware_version": drive.get('firmware_version'),
            "capacity_bytes": drive.get('capacity_bytes'),
            "drive_type": drive.get('drive_type'),
            "interface": drive.get('interface'),
        }

        try:
            smart_data = self._get_smart_data_from_device_obj(device_obj)
            data.update(smart_data)
        except Exception as e:
            smart_attributes = self._get_empty_smart_attributes()
            data.update(smart_attributes)

        return data

    def _get_smart_data_from_device_obj(self, device: PySMART_Device) -> Dict[str, Any]:
        """Получение SMART данных из объекта pySMART Device"""
        smart_data: Dict[str, Any] = {}
        device.update()

        smart_data['smart_status'] = device.assessment or 'Unknown'

        attributes = self._get_empty_smart_attributes()

        if device.if_attributes is not None:
            self._extract_common_attributes(device.if_attributes, attributes)

            if isinstance(device.if_attributes, AtaAttributes):
                self._extract_ata_attributes(device.if_attributes, attributes)
            elif isinstance(device.if_attributes, NvmeAttributes):
                self._extract_nvme_attributes(device.if_attributes, attributes)

        smart_data.update(attributes)
        return smart_data

    def _get_smart_data_from_name(self, device_name: str) -> Dict[str, Any]:
        """Fallback: получение SMART данных через создание нового Device объекта"""
        try:
            device = PySMART_Device(device_name)
            return self._get_smart_data_from_device_obj(device)
        except Exception as e:
            print(f"Ошибка создания Device для {device_name}: {e}")
            return self._get_empty_smart_attributes()

    def _extract_common_attributes(self, if_attrs, attributes: Dict[str, Any]) -> None:
        """Извлечение общих атрибутов из любого интерфейса"""
        if isinstance(if_attrs, AtaAttributes):
            if not if_attrs.legacyAttributes or all(attr is None for attr in if_attrs.legacyAttributes):
                return
            for attr in if_attrs.legacyAttributes:
                if attr is not None:
                    attr_name = attr.name.lower().replace('-', '_').replace(' ', '_')
                    if attr_name == 'temperature_celsius':
                        attributes['temperature'] = attr.value_int
                    elif attr_name == 'power_on_hours':
                        attributes['power_on_hours'] = attr.value_int
                    elif attr_name == 'power_cycle_count':
                        attributes['power_cycles'] = attr.value_int

        elif isinstance(if_attrs, NvmeAttributes):
            attributes['temperature'] = if_attrs.temperature
            attributes['power_on_hours'] = if_attrs.powerOnHours
            attributes['power_cycles'] = if_attrs.powerCycles

        elif isinstance(if_attrs, SCSIAttributes):
            attributes['temperature'] = if_attrs.temperature

    def _extract_ata_attributes(self, ata_attrs: AtaAttributes, attributes: Dict[str, Any]) -> None:
        """Извлечение специфичных ATA атрибутов"""
        if not ata_attrs.legacyAttributes or all(attr is None for attr in ata_attrs.legacyAttributes):
            return

        for attr in ata_attrs.legacyAttributes:
            if attr is not None:
                attr_name = attr.name.lower().replace('-', '_').replace(' ', '_')
                attributes[f'ata_{attr_name}_raw'] = attr.raw
                attributes[f'ata_{attr_name}_normalized'] = attr.value_int

    def _extract_nvme_attributes(self, nvme_attrs: NvmeAttributes, attributes: Dict[str, Any]) -> None:
        """Извлечение специфичных NVMe атрибутов"""
        attributes['nvme_percentage_used'] = nvme_attrs.percentageUsed
        attributes['nvme_data_units_written'] = nvme_attrs.dataUnitsWritten
        attributes['nvme_data_units_read'] = nvme_attrs.dataUnitsRead
        attributes['nvme_integrity_errors'] = nvme_attrs.integrityErrors
        attributes['nvme_unsafe_shutdowns'] = nvme_attrs.unsafeShutdowns
        attributes['nvme_available_spare'] = nvme_attrs.availableSpare
        attributes['nvme_critical_warning'] = nvme_attrs.criticalWarning
