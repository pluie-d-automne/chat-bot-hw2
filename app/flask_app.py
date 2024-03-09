from flask import Flask, render_template, request
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Загрузка модели
model_id = "TheBloke/Llama-2-7B-Chat-fp16"
# Load the entire model on the GPU 0
device_map = {"": 0}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    quantization_config=bnb_config
)
llm_model = PeftModel.from_pretrained(base_model, './app/static/weights')
llm_model = llm_model.merge_and_unload()

# Reload tokenizer to save it
llm_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = "right"

# Функция для получения промпта из контакста диалога
def generate_prompt(context):
    prompt_template = """You are Homer Simpson. Add one answer to the dialog below.\n\n{query}\n\n### Answer: """
    query = '\n\n'.join(context)
    prompt = prompt_template.format(query=query)
    return prompt

def get_completion(prompt: str, model, tokenizer) -> str:
  device = "cuda"
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)
  generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  return (decoded[0])

context = list()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('base.html')

@app.route("/get")
def get_homer_response():
    query = request.args.get('msg')
    context.append("### Someone: "+query)
    context = context[-10:] if len(context)>10 else context
    result = get_completion(prompt=generate_prompt(context), model=llm_model, tokenizer=llm_tokenizer)
    result = result[result.index('### Answer:'):].replace('### Answer:', '### Homer Simpson:')
    context.append(result)
    return result.replace('### Homer Simpson:', '')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')