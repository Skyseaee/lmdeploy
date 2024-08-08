#Large Languase Model, such as llama2-7b.
from calflops import calculate_flops
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

batch_size = 1
max_seq_length = 1024
model_path = "/workspace/yakun/llm/vicuna-13b-v1.5"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=(batch_size, max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("Llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
