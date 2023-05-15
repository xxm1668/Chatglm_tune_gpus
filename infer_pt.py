import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("./output_pt", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./output_pt", trust_remote_code=True)

model = model.eval()

input_text = '一个患者的卵巢小细胞癌转移至其它部位，是否有必要进行手术治疗？'
input_text = tokenizer.tokenize(input_text)
input_text = input_text + ["[gMASK]", "<sop>"]
ids = tokenizer.convert_tokens_to_ids(input_text)
print(ids)
input_ids = torch.LongTensor([ids]).cuda()
print(input_ids)
generation_kwargs = {
    "min_length": 10,
    "max_new_tokens": 150,
    "top_p": 0.7,
    "temperature": 0.95,
    "do_sample": False,
    "num_return_sequences": 1,
}

with torch.no_grad():
    out = model.generate(
        input_ids=input_ids, **generation_kwargs
    )
out_text = tokenizer.decode(out[0])
print(out_text)
