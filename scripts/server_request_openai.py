################################################################################
# @Copyright: 2019-2024 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2024-03-20 18:02:51
# @Details  :
# @Update   : add v1/chat/completions client code for VLM model
################################################################################

from typing import Any, Dict, Iterable, List, Optional, Union
import json
import requests
import logging

# Credit to: https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
def json_loads(content):
    """Loads content to json format."""
    try:
        content = json.loads(content)
        return content
    except:  # noqa
        logging.warning(f'weird json content {content}')
        return ''


class APIClient:
    def __init__(self, server_addr: str, enable_chat=False):
        self.completions_chat_v1_url = f'{server_addr}/v1/chat/completions'
        self.completions_v1_url = f'{server_addr}/v1/completions'
        self._models_v1_url = f'{server_addr}/v1/models'
        self.model_name = self.get_model_list(self._models_v1_url)[0]
        self.enable_chat = enable_chat
        print(f"model_name: {self.model_name}")

    @staticmethod
    def get_model_list(api_url: str):
        """Get model list from api server."""
        response = requests.get(api_url)
        if hasattr(response, 'text'):
            model_list = json.loads(response.text)
            model_list = model_list.pop('data', [])
            return [item['id'] for item in model_list]
        return None
    
    def completions_v1(
            self,
            prompt: Union[str, List[Any]],
            suffix: Optional[str] = None,
            temperature: Optional[float] = 0.7,
            n: Optional[int] = 1,
            max_tokens: Optional[int] = 16,
            stream: Optional[bool] = False,
            top_p: Optional[float] = 1.0,
            top_k: Optional[int] = 40,
            user: Optional[str] = None,
            # additional argument of lmdeploy
            repetition_penalty: Optional[float] = 1.0,
            session_id: Optional[int] = -1,
            ignore_eos: Optional[bool] = False,
            traceid: Optional[Union[str, int]] = 12121212,
            **kwargs):
        """Chat completion v1.

        Args:
            model (str): model name. Available from /v1/models.
            prompt (str): the input prompt.
            suffix (str): The suffix that comes after a completion of inserted
                text.
            max_tokens (int): output token nums
            temperature (float): to modulate the next token probability
            top_p (float): If set to float < 1, only the smallest set of most
                probable tokens with probabilities that add up to top_p or
                higher are kept for generation.
            top_k (int): The number of the highest probability vocabulary
                tokens to keep for top-k-filtering
            n (int): How many chat completion choices to generate for each
                input message. Only support one here.
            stream: whether to stream the results or not. Default to false.
            repetition_penalty (float): The parameter for repetition penalty.
                1.0 means no penalty
            user (str): A unique identifier representing your end-user.
            ignore_eos (bool): indicator for ignoring eos
            session_id (int): if not specified, will set random value

        Yields:
            json objects in openai formats
        """
        pload = {
            'model': self.model_name,
            'stream': stream,
            'max_tokens': max_tokens,
            'top_k': top_k,
            'top_p': top_p,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'traceid': traceid
        }
        if self.enable_chat:
            pload['messages'] = prompt 
        else:
            pload['prompt'] = prompt

        headers = {'content-type': 'application/json'}
        print(pload)
        response = requests.post(self.completions_chat_v1_url if self.enable_chat else self.completions_v1_url,
                                 headers=headers,
                                 json=pload,
                                 stream=stream)
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b'\n'):
            if chunk:
                if stream:
                    decoded = chunk.decode('utf-8')
                    if decoded == 'data: [DONE]':
                        continue
                    if decoded[:6] == 'data: ':
                        decoded = decoded[6:]
                    output = json_loads(decoded)
                    yield output
                else:
                    decoded = chunk.decode('utf-8')
                    output = json_loads(decoded)
                    yield output

import base64

def encode_image_to_base64(image_path):
    """将本地图片编码为 Base64 格式"""
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_encoded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-addr",
        type=str,
        default='http://sg12.aip.mlp.shopee.io/services/618001',
        help="Parse path of helm, you need locate in benchmark_output directory",
    )
    parser.add_argument(
        "--enable-chat",
        action='store_true',
        help="request /v1/chat/completions",
    )
    parser.add_argument(
        "--disable-stream",
        action='store_true',
        help="stream response",
    )
    args = parser.parse_args()
    # vlm request use chat interface
    api_client = APIClient(args.server_addr, args.enable_chat)
    request_output_len = 256
    top_k = 3
    top_p = 0.95
    temperature = 0.7
    repetition_penalty = 1.15
    if args.enable_chat:
        # for MiniCPM-Llama3-V2.5
        temperature = 0.8
        top_p = 0.8
        top_k = 40
        repetition_penalty = 1.0
    
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': "Please extract the size table information contained in the picture. The following are the extraction rules: 1.Only the size table information in the picture needs to be extracted. 2.Size information does not need to be translated into Chinese or English. 3.Don't create your own words and sizes. 4.The results are returned in markdown format. 5.Please try to ensure that the returned results are consistent with the table in the picture. 6.If not, please return empty.",
        }, {
            'type': 'image_url',
            'image_url': {
                'url': "https://cf.shopee.co.id/file/43f94516eb828fe2fbbc6badfb476bf6",
            },
        }],
    }]
    # image_path = "test_data/Seamoney_MY_PAYSLIP/1986620857220355073.jpg"
    # base64_image = encode_image_to_base64(image_path)


    # messages = [
    # {'role': 'system', 'content': "You are now in the role of an expert credit reviewer who can identify employee pay slip. \nYou follow instructions precisely and without deviation."},
    # {
    #     'role': 'user',
    #     'content': [
    #     {
    #         'type': 'text',
    #         'text': "逐步思考以检查提供的图片是否是工资单。一个合格的工资单一定要包含用户的薪资信息（如 basic pay / gross pay / net pay / jumlah pendapatan / gaji bersih 之一或几项）等信息\n注意：银行交易转账记录不被视为工资单。\n\n您必须仅以以下结构的JSON模式进行回复。不要添加任何额外的评论。如果上传的图像是工资单，则返回true，否则返回false：\n{\"is_payslip\": true/false}",
    #     }, {
    #         'type': 'image_base64',
    #         'image_base64': {
    #             'data': base64_image,
    #         },
    #     }],
    # }]

    if not args.disable_stream:
        for result in api_client.completions_v1(
            prompt=messages if args.enable_chat else "Hello, Please tell me some details about yourself",
            max_tokens=request_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stream=not args.disable_stream):
            if args.enable_chat:
                if result['choices'][0]['delta']['content']: 
                    print(result['choices'][0]['delta']['content'], end='', flush=True)
            else:
                if result['choices'][0]['text']: 
                    print(result['choices'][0]['text'], end='', flush=True)
        print("\n")
    else:
        for output in api_client.completions_v1(
            prompt=messages if args.enable_chat else "Hello, Please tell me some details about yourself",
            max_tokens=request_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stream=not args.disable_stream):
            print(output)
