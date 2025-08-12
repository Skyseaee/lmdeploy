################################################################################
# @Copyright: 2019-2024 Shopee. All Rights Reserved.
# @Author   : wenlong.cao@shopee.com
# @Date     : 2024-06-18 15:00:00
# @Details  :
################################################################################
# pip3 install openai
from openai import OpenAI

client = OpenAI(api_key='EMPTY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id

def request_vlm(url, stream=False):
    res = client.chat.completions.create(
        model=model_name,
        messages=[{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': "Please extract the size table information contained in the picture. The following are the extraction rules: 1.Only the size table information in the picture needs to be extracted. 2.Size information does not need to be translated into Chinese or English. 3.Don't create your own words and sizes. 4.The results are returned in markdown format. 5.Please try to ensure that the returned results are consistent with the table in the picture. 6.If not, please return empty.",
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': url,
                },
            }],
        }],
        temperature=0.8,
        top_p=0.8,
        stream=stream)
    if stream:
        for chunk in res:
            out = chunk.choices[0].delta.content
            usage = dict(chunk.usage) if chunk.usage else chunk.usage
            finish_reason = chunk.choices[0].finish_reason
            yield out, usage, finish_reason
    else:
        out = res.choices[0].message.content
        usage = dict(res.usage)
        finish_reason = res.choices[0].finish_reason
        yield out, usage, finish_reason 

urls = ["https://cf.shopee.co.id/file/43f94516eb828fe2fbbc6badfb476bf6",
        "https://cf.shopee.co.id/file/id-11134207-7r98p-loi9q7w32cu13e"]

stream = True
for url in urls:
    for out, usage, finish_reason in request_vlm(url, stream):
        if stream:
            print(out or '', end='', flush=True)
        else:
            print(out)
    print()