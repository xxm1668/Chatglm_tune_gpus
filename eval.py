import json
from metrics import compute_metrics

ori_list, gen_list = [], []
examples = []
with open("/Users/haojingkun/Downloads/ChatGLM-Tuning2/predictions.json", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        examples.append(json.loads(line))
    # examples = json.loads(f.read())
    for example in examples:
        ori_list.append(example['ori_answer'])
        try:
            gen_list.append(example['gen_answer'].split('\nAnswer: ')[1])
        except:
            gen_list.append(example['gen_answer'])
    score_dict = compute_metrics(ori_list, gen_list)
    print(json.dumps(score_dict, ensure_ascii=False, indent=4))
