from core.data_storage import FileSystemDataProvider


DICT_DATA_PROVIDERS = {
    "fs": FileSystemDataProvider,
}
DEFAULT_PROVIDER_TYPE = "fs"
DEFAULT_PROVIDER_PARAMS = {
    "fs": {
        "base_dir": "storage/data",
    }
}
