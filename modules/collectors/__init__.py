from .cpu_collector import CpuCollectorMacOS, CpuCollectorLinux
from .gpu_collector import GpuCollectorLinux
from .drive_collector import DriveCollectorLinux
from .battery_collector import BatteryCollector

DICT_COLLECTORS = {
    "Darwin": {
        "cpu": CpuCollectorMacOS,
        "battery": BatteryCollector
    },
    "Windows": {
        "battery": BatteryCollector
    },
    "Linux": {
        "cpu": CpuCollectorLinux,
        "gpu": GpuCollectorLinux,
        "drive": DriveCollectorLinux,
        "battery": BatteryCollector
    }
}
