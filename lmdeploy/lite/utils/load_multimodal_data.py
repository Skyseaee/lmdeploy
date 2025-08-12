import os
from typing import List, Tuple, Union
from PIL import Image
from tqdm import tqdm


def check_image_folder(folder_path: str) -> bool:
    # chech if the folder fath exists
    if os.path.isdir(folder_path):
        return True
    else:
        return False
    
def load_wikimedia_data(dataset: str = 'wikimedia/wit_base'):
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", streaming=True)
    text, image = [], []
    count = 0
    max_count = 1000

    with tqdm(total=max_count, desc="Loading wikimedia/wit_base data") as pbar:
        for sample in ds:
            if count >= max_count:
                break
            
            language = sample['wit_features']['language']
            if 'en' in language:
                index = language.index('en')
            else:
                continue

            current_text = sample['wit_features']['context_page_description'][index]
            if not current_text or len(current_text) == 0:
                continue
            
            text.append(current_text)
            image.append(sample['image'])
            count += 1

            pbar.update(1)

    return text, image


def load_multimodal_data(dataset_path: str, prompt_file: str, image_folder: str = 'images', ais_dataset=False, ais_key='') -> Tuple[List[str], List[Union[str, Image.Image]]]:
    if dataset_path == 'wikimedia/wit_base':
        text, images = load_wikimedia_data()
        return text, images

    file_format = prompt_file.split('.')[-1]

    if ais_dataset:
        print('[INFO] Trying to load multimodal dataset from AIS...')
        try:
            dataset_path = download_multimodal_data_from_ais(dataset_path, ais_key)
            print(f'[INFO] Load AIS dataset successfully, path is {dataset_path}.')
        except Exception:
            print('[ERROR] Load dataset from AIS failed. Please check the token / dataset_id / commit_id')
            raise

    file_path = os.path.join(dataset_path, prompt_file) if not ais_dataset else os.path.join(dataset_path, 's3upload', prompt_file)
    texts, images = [], []

    folder_path = os.path.join(dataset_path, image_folder) if not ais_dataset else os.path.join(dataset_path, 's3upload', image_folder)
    assert check_image_folder(folder_path), (
        f"The folder '{folder_path}' does not exist. "
        "Please use the `--image-folder` flag to specify the correct folder path "
        "or rename your folder to match the expected name 'images'."
    )

    if file_format == 'jsonl':
        import json

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)

                text = data.get('text', '')
                image_name = data.get('image', '')
                
                texts.append(text)
                if image_name.startswith('http'):
                    images.append(image_name)
                else:
                    images.append(os.path.join(folder_path, image_name))
        
        return texts, images
    elif file_format == 'csv':
        import csv

        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                image_name = row['image']

                texts.append(text)
                if image_name.startswith('http'):
                    images.append(image_name)
                else:
                    images.append(os.path.join(folder_path, image_name))

        return texts, images
    elif file_format == 'json':
        import json

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)['data']
            for group in data:
                text = group['text']
                image_name = group['image']

                texts.append(text)
                if image_name.startswith('http'):
                    images.append(image_name)
                else:
                    images.append(os.path.join(folder_path, image_name))

        return texts, images
    else:
        supported_formats = ['jsonl', 'csv', 'json']

        raise Exception(
            f"The file format '{file_format}' is not supported yet. "
            f"Please reorganize your multimodal file in one of the supported formats: {supported_formats}."
        )

def load_data_from_ais(dataset_id: int, dataset_commit: str, token: str, split: str="calibrate", cache_dir:str="~/.datasets"):
    """
    load image from ais
    """
    print(f'[INFO] loading {dataset_id}-{dataset_commit} from AIS Dataset')

    from datatoolchain import load_from_hub, load_from_local

    cache_dir = os.path.expanduser(cache_dir)

    if os.path.exists(cache_dir):
        return load_from_local(local_path=cache_dir)

    dataset = load_from_hub(
        dataset_id=dataset_id,
        commit_id=dataset_commit,
        cache_dir=cache_dir,
        split=split,
        use_auth_token=token
    )

    return dataset

def load_storage_dataset(token: str, dataset_id: int, dataset_commit: str):
    print(f'[INFO] loading {dataset_id}-{dataset_commit} from AIS Storage Dataset')
    os.environ['AIS_TOKEN'] = token

    from datatoolchain import storage

    sto = storage.DatasetStorage.create_io_storage(dataset_id=dataset_id, commit_id=dataset_commit)
    return sto

def download_files(sto, dataset_info: str, cache_dir='.dataset') -> str:
    path = os.path.join(cache_dir, dataset_info)

    if os.path.exists(path):
        return path
    else:
        # print(path, filename)
        sto._cache.download(path)
        return path

def download_multimodal_data_from_ais(dataset_path: str, ais_key: str):
    """
        This method will download the image from AIS in cache_dir and replace the image path.
    """
    token = os.getenv("AIP_TOKEN", '')
    if len(token) == 0:
        print('[ERROR] No AIP_TOKEN found, \
              please set the AIP_TOKEN in your environment by running the command export AIP_TOKEN=your_token')
        return
    
    dataset_id = os.getenv("AIP_DATASET_ID", '3746')
    dataset_commit = os.getenv("AIP_DATASET_COMMIT_ID", 'a50f6734')
    if len(dataset_id) == '' and len(dataset_commit) == '':
        datasets = dataset_path.split('_')
        dataset_id, dataset_commit = datasets[0], dataset_id[1]
    
    dataset = load_storage_dataset(token, int(dataset_id), dataset_commit)
    return download_files(dataset, str(dataset_id) + dataset_commit)
