from .cpu_collector import CpuCollectorMacOS, CpuCollectorLinux
from .gpu_collector import GpuCollectorLinux
from .drive_collector import DriveCollectorLinux

DICT_COLLECTORS = {
    "Darwin": {
        "cpu": CpuCollectorMacOS,
    },
    "Windows": {
    },
    "Linux": {
        "cpu": CpuCollectorLinux,
        "gpu": GpuCollectorLinux,
        "drive": DriveCollectorLinux
    }
}
