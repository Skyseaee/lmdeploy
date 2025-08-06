# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Union
from lmdeploy import messages
from lmdeploy.model import BaseChatTemplate
import numpy as np
import torch
from PIL import Image

from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.lite.utils.load_multimodal_data import load_multimodal_data
from lmdeploy.vl.utils import load_image


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(tokenizer, nsamples, seed, seqlen):
    """Load Wikitext-2 train and test datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized Wikitext-2 test set.
    """
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)

    trainenc = tokenizer('\n\n'.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer('\n\n'.join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(tokenizer, nsamples, seed, seqlen):
    """Load PTB train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True)

    trainenc = tokenizer('\n\n'.join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer('\n\n'.join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(tokenizer, nsamples, seed, seqlen):
    """Load C4 train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4',
                             'allenai--c4',
                             data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                             split='train',
                             use_auth_token=False,
                             trust_remote_code=True)
    valdata = load_dataset('allenai/c4',
                           'allenai--c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation',
                           use_auth_token=False,
                           trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(tokenizer, nsamples, seed, seqlen):
    """Load PTB New train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)

    trainenc = tokenizer(' '.join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(' '.join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(tokenizer, nsamples, seed, seqlen):
    """Load C4 New train and validation datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4',
                             'allenai--c4',
                             data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                             split='train',
                             trust_remote_code=True)
    valdata = load_dataset('allenai/c4',
                           'allenai--c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation',
                           trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_pileval(tokenizer, nsamples, seed, seqlen=512):
    """Load pileval train dataset and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
        test_enc: Full tokenized PTB validation set.
    """
    from datasets import load_dataset
    from datasets.builder import DatasetGenerationError
    try:
        dataset = load_dataset('mit-han-lab/pile-val-backup', split='validation', trust_remote_code=True)
    except DatasetGenerationError:
        raise InterruptedError('There have been some issues when generating '
                               'the dataset, you could try to download it '
                               'locally first, and replace the `data_files`'
                               'with local addresses or use other datasets '
                               '(c4, wiki, ptb).')
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == nsamples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    print(f' * Split into {n_split} blocks')
    return [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)], None


def get_ultrachat_2k(tokenizer, nsamples, seed, seqlen):
    """Load ultrachat_2k train_sft datasets and tokenize.

    Args:
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from train set.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        train_loader: List of sampled and tokenized training examples.
    """
    from datasets import load_dataset
    traindata = load_dataset("mgoin/ultrachat_2k", split="train_sft")
    trainenc = tokenizer('\n\n'.join(traindata['prompt']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


async def get_llvm_dataset(tokenizer, nsamples, seed, seqlen=2048, **kwargs):
    """Load llvm dataset and tokenize for llvm.

    Args:
        dataset_path: a string of path to a local dataset file path.
        tokenizer: Tokenizer to encode text.
        nsamples: Number of samples to take from dataset.
        seed: Random seed for sampling.
        seqlen: Maximum sequence length.

    Returns:
        [samples]: List of samples with torch.Tensor.
        None
    """
    from lmdeploy.messages import TurbomindEngineConfig
    
    dataset_path = kwargs['dataset_path'] if 'dataset_path' in kwargs else 'wikimedia/wit_base'
    model_path = kwargs['model_path'] 
    prompt_file = kwargs['prompt_file'] if 'prompt_file' in kwargs else ''
    ais_dataset = True if 'ais_dataset' in kwargs else False
    ais_key = kwargs['ais_key'] if 'ais_key' in kwargs else ''

    image_folder = kwargs['image_folder'] if 'image_folder' in kwargs else 'images'

    texts, images = load_multimodal_data(dataset_path, prompt_file, image_folder, ais_dataset, ais_key)

    combined = list(zip(texts, images))
    random.seed(seed)
    np.random.seed(seed=seed)
    random.shuffle(combined)

    vl_encoder = ImageEncoder(model_path, backend='turbomind', backend_config=TurbomindEngineConfig(tp=1))
    
    samples = []
    temp = {'input_ids': None, 'input_embeddings': None} 
    n_run = 0
    
    # for text, image in combined:
    index = 0
    while n_run < nsamples:
        text, image = combined[index]
        
        index += 1
        line = text.strip()
        
        # encode image
        result = await encoding_image(image, line, tokenizer, vl_encoder)

        if temp['input_ids'] is None:
            temp['input_ids'] = result['input_ids']
            temp['input_embeddings'] = result['input_embeddings']
        else:
            temp['input_ids'].extend(result['input_ids'])
            temp['input_embeddings'] = torch.cat((temp['input_embeddings'], result['input_embeddings']), dim=0)

        if len(temp['input_ids']) > seqlen:
            i = np.random.randint(0, len(temp['input_ids']) - seqlen)
            j = i + seqlen
            sample = torch.tensor([temp['input_ids'][i:j]])
            embeddings = torch.unsqueeze(temp['input_embeddings'][i:j, :], dim=0)

            if sample.numel() == 0:
                continue

            samples.append((sample, embeddings))
            n_run += 1
            temp = {'input_ids': None, 'input_embeddings': None} 

    print(f'All sample number is {len(samples)}\n')
    return samples, None   


async def encoding_image(image_path: Union[str, Image.Image], prompt: str, tokenizer, vl_encoder):
    if isinstance(image_path, str):
        image_path = load_image(image_path)
        item = {'type': 'image', 'image': image_path}
    elif isinstance(image_path, Image.Image):
        item = {'type': 'image', 'image': image_path}
    else:
        raise ValueError(f'Invalid image path: {image_path}')

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                item,
            ]
        }
    ]

    results = await vl_encoder.preprocess(messages)
    results = await vl_encoder.async_infer(results)
    
    # tm_results = await vl_encoder.wrap_for_pytorch(results, chat_template, tokenizer, sequence_start=True)
    features = next(m['content'][0] for m in results if m['role'] == 'forward')
    features = features.cpu()
    # input_ids = tm_results['input_ids']

    segs = [
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:", 
        prompt
    ]

    final_embeddings = []
    final_input_ids = []
    embedding_ranges = []

    for i, seg in enumerate(segs):
        # add text seg
        seg_ids = tokenizer.encode(seg)
        final_input_ids.extend(seg_ids)
        final_embeddings.append(torch.zeros(len(seg_ids), features.size(1)))

        if i == 0:
            img_start = len(final_input_ids)
            final_input_ids.extend([0] * len(features))
            final_embeddings.append(features)
            img_end = len(final_input_ids)
            embedding_ranges.append((img_start, img_end))
        

    return {
        'input_embeddings': torch.cat(final_embeddings, dim=0),
        'input_ids': final_input_ids,
        'embedding_ranges': embedding_ranges,
    }


def get_calib_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048, **kwargs):
    """Get calibration data loaders for a dataset.

    Args:
      name: Dataset name ('wikitext2', 'ptb', 'c4', etc).
      tokenizer: Tokenizer to encode text.
      nsamples: Number of samples to take from train set.
      seed: Random seed for sampling.
      seqlen: Maximum sequence length.

    Returns:
      train_loader: List of sampled and tokenized training examples.
      test_data: Full tokenized validation set.
    """
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, nsamples, seed, seqlen)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(tokenizer, nsamples, seed, seqlen)
        return get_ptb(tokenizer, nsamples, seed, seqlen)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer, nsamples, seed, seqlen)
        return get_c4(tokenizer, nsamples, seed, seqlen)

    if 'pileval' in name:
        return get_pileval(tokenizer, nsamples, seed, seqlen)

    if 'ultrachat_2k' in name:
        return get_ultrachat_2k(tokenizer, nsamples, seed, seqlen)
    
    if 'llvm' in name:
        async def _async_wrapper():
            return await get_llvm_dataset(tokenizer, nsamples, seed, seqlen, **kwargs)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_async_wrapper())
