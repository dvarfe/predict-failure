from .fs_provider import FileSystemDataProvider
from .global_fs_provider import GlobalFileSystemProvider


DICT_DATA_PROVIDERS = {
    "fs": FileSystemDataProvider,
    "global_fs": GlobalFileSystemProvider,
}
DEFAULT_PROVIDER_TYPE = "fs"
DEFAULT_PROVIDER_PARAMS = {
    "fs": {
        "base_dir": "storage/data",
    }
}

