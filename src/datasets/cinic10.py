import os

import requests
import shutil
import tarfile

import torchvision
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from src.utils.exception import DatasetValidationException


class CINIC10:
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
    default_url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
    dataset_path_map = {'train': 'train',
                        'test': 'test',
                        'validation': 'valid'}

    def __init__(self, root='.', download=True, url=None):

        self.url = CINIC10.default_url if url is None else url
        self.path = root
        if download:
            if not self._dataset_exists():
                self._remove_leftover()
                self._download_set()
        if not self._dataset_exists():
            raise DatasetValidationException('Dataset is either not in path or invalid.')

    def _extract_dataset(self, file_name):
        with tarfile.open(os.path.join(self.path, file_name)) as file:
            file.extractall(self.path)

    def _dataset_exists(self):
        folders_exist = True
        for folder in CINIC10.dataset_path_map.values():
            folders_exist = os.path.exists(os.path.join(self.path, folder)) and folders_exist
        return folders_exist

    def _remove_leftover(self):
        for folder in CINIC10.dataset_path_map.values():
            complete_path = os.path.join(self.path, folder)
            if os.path.exists(complete_path):
                shutil.rmtree(complete_path)

    def _download_set(self):
        tqdm.write("Starting downlod of CINIC-10 dataset...")
        with requests.get(self.url, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            file_path = os.path.join(self.path, os.path.basename(r.url))
            file_name = os.path.basename(r.url)
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(f"{file_path}", 'wb') as output:
                    shutil.copyfileobj(raw, output)

        tqdm.write("Download of CINIC-10 dataset finished.")
        tqdm.write("Extracting Dataset...")
        self._extract_dataset(file_name)
        tqdm.write("Extraction of Dataset finished.")
        os.remove(file_path)

    def get_dataset(self, subset: str):
        if subset not in CINIC10.dataset_path_map.keys():
            raise ValueError("Invalid subset name")
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=CINIC10.mean, std=CINIC10.std)])
        return torchvision.datasets.ImageFolder(os.path.join(self.path, CINIC10.dataset_path_map[subset]),
                                                transform=transform)