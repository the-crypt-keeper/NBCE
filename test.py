#! -*- coding: utf-8 -*-
# Naive Bayes-based Context Extension (NBCE)
# Use Naive Bayes to increase the length of LLM's Context processing
# Link: https://kexue.fm/archives/9617
# Torch 2.0 test passed

import json
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import TopPLogitsWarper, LogitsProcessorList

# Fine-tuned LLAMA
# Download address: https://openbuddy.ai/
model_path = '/root/autodl-tmp/7b-trans-chat-0516-bf16'

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left' 
tokenizer.pad_token = tokenizer.unk_token

# Load the LLAMA model
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
device = torch.device('cuda')

# load example Context
contexts = json.load(open('contexts.json'))

# Example question set (ask multiple questions at once, NBCE will output answers one by one according to the context)
question = """Please read the materials carefully and answer one by one:
- National Grid Corporation of the Philippines, how much is China's share?
- How many people does LinkedIn plan to lay off?
- How much did Gilead pay for Pharmasset?
- In what year was Sovaldi, the miracle drug for hepatitis C, launched?
- Where will the Central Asia Summit be held? Hosted by?
- Which actor was investigated for insulting the People's Army?
- Which project claims a "tank-able" waterway?
- If you were the CEO of Merck, what would be your top priority? """

# splicing context and question
contexts = [''] + contexts # Add an empty Context (no Context prediction)
batch = ['User: %s\n\n%s\n\nAssistant:' % (context, question) for context in contexts]
print('Context length distribution:', [len(text) for text in batch])
print('Context total length:', sum([len(text) for text in batch]))

# Top-P truncation
processors = LogitsProcessorList()
processors.append(TopPLogitsWarper(0.95))


@torch.inference_mode()
def generate(max_tokens):
    """Naive Bayes-based Context Extension demo code
    """
    inputs = tokenizer(batch, padding='longest', return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    print('input_ids', input_ids.shape)
    past_key_values = None
    n = input_ids.shape[0]
    
    for i in range(max_tokens):
        # Model output
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values
                       )
        past_key_values = outputs.past_key_values
        
        # ===== Core Code Start =====
        beta = 0.25
        logits = outputs.logits[:, -1]
        logits = logits - logits.logsumexp(dim=-1, keepdims=True)
        logits = processors(input_ids, logits)
        k = (logits.exp() * logits.clip(-100, 0)).sum(dim=-1)[1:].argmax() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits_merged = (1 + beta) * logits_max - beta * logits_uncond
        logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
        # ===== End of core code =====
        
        # build distribution, sample
        # tau = 1 is standard random sampling, tau->0 is greedy search
        # For simplicity, topk and topp truncation are not implemented here
        tau = 0.01
        probas = torch.nn.functional.softmax(logits[None] / tau , dim=-1)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)        
        if next_tokens[0] == tokenizer.eos_token_id:
            break
            
        ret = tokenizer.batch_decode(next_tokens)
        print(ret[0], flush=True, end='')
        
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)        


if __name__ == '__main__':
    generate(1000)


"""
========= Output result reference =========

1. How much is China's shareholding in National Grid Corporation of the Philippines?
Answer: The State Grid Corporation of China holds 40% of the shares of the National Grid Corporation of the Philippines.

2. How many people does LinkedIn plan to lay off?
A: LinkedIn plans to lay off 716 people.

3. How much did Gilead pay for Pharmasset?
A: Gilead acquired Pharmasset for $11 billion.

4. In what year was Sovaldi, the miracle drug for hepatitis C, launched?
Answer: Sovaldi, a miracle drug for hepatitis C, was launched in 2013.

5. Where will the Central Asia Summit be held? Hosted by?
A: The Central Asia Summit will be held in Xi'an, Shaanxi Province and will be hosted by President Xi Jinping.

6. Which actor was investigated for insulting the People's Army?
A: Li Haoshi was investigated for insulting the People's Army during his performance.

7. Which project claims to be a "tank-able" water road?
Answer: Enshi, Hubei proclaimed that the "tank-passing" water road.

8. If you were the CEO of Merck, what would be your top priority?
Answer: If I were the CEO of Merck, my first task would be how to make the basic market stronger and achieve better growth through drug combination.
"""