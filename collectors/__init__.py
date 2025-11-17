from .cpu_collector import CpuCollectorMacOS, CpuCollectorLinux
from .gpu_collector import GpuCollectorLinux

DICT_COLLECTORS = {
    "Darwin": {
        "cpu": CpuCollectorMacOS,
    },
    "Windows": {
    },
    "Linux": {
        "cpu": CpuCollectorLinux,
        "gpu": GpuCollectorLinux
    }
}
