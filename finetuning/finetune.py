import sys
import logging

import datasets
from datasets import load_dataset
from peft import LoraConfig, PeftModel
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import json

"""
Adapted for navigation task finetuning
"""

logger = logging.getLogger(__name__)


###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": True,
    "learning_rate": 5.0e-05,
    "log_level": "info",
    "logging_steps": 50, 
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 5, 
    "max_steps": -1,
    "output_dir": "./phi3-mr-lora-fixed-v3", 
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 2,
    "per_device_train_batch_size": 2,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 2, 
    "seed": 42, 
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.05, 
    "eval_strategy": "steps",
    "eval_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "max_grad_norm": 0.5,
    "weight_decay": 0.1,
    }

peft_config = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.1,
    "bias": "all",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
        "embed_tokens",  # ← AGGIUNTO: embeddings
        "lm_head"        # ← AGGIUNTO: output layer
    ],
    "modules_to_save": ["embed_tokens", "lm_head"],  # ← salva completamente questi layer
}
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)


###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")


################
# Model Loading
################
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="eager",  # MODIFICATO: da "flash_attention_2" a "eager" (non hai flash attention)
    torch_dtype=torch.bfloat16,
    device_map="auto"  # MODIFICATO: da None a "auto"
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.eos_token  # MODIFICATO: da unk_token a eos_token (fix critico)
tokenizer.pad_token_id = tokenizer.eos_token_id  # MODIFICATO: usa eos_token_id
tokenizer.padding_side = 'right'

########################################
#        DATASET PRE-CHECKS            #
########################################

with open("finetune_data.jsonl", 'r') as f:
    samples = [json.loads(line) for line in f]

# Check 1: Tutti hanno assistant response?
for i, s in enumerate(samples[:5]):
    msgs = s["messages"]
    if len(msgs) != 3 or msgs[2]["role"] != "assistant":
        print(f"❌ Sample {i} malformato!")
    else:
        print(f"✅ Sample {i} OK - response: {msgs[2]['content'][:50]}...")

# Check 2: Diversity
user_prompts = [s["messages"][1]["content"] for s in samples]
unique_starts = len(set(p[:100] for p in user_prompts))
print(f"\n📊 Diversity: {unique_starts}/{len(samples)} unique prompt starts")
if unique_starts < len(samples) * 0.7:
    print("⚠️ Warning: Low diversity, many similar samples")


##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)  # MODIFICATO: da False a True (fix critico!)
    return example

# MODIFICATO: cambiato dataset
raw_dataset = load_dataset("json", data_files={"train": "finetune_data.jsonl"})
train_test = raw_dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=1,  # MODIFICATO: da 10 a 1 (più sicuro con dataset piccolo)
    remove_columns=column_names,
    desc="Applying chat template to train",
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=1,  # MODIFICATO: da 10 a 1
    remove_columns=column_names,
    desc="Applying chat template to test",
)


###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    processing_class=tokenizer,
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


############
# Save model
############
trainer.save_model(train_conf.output_dir)


########################################
# AGGIUNTO: INFERENCE TEST
########################################
logger.info("\n" + "=" * 80)
logger.info("Running inference tests...")
logger.info("=" * 80)

# Load finetuned model
test_model = PeftModel.from_pretrained(
    model,
    train_conf.output_dir,
    torch_dtype=torch.bfloat16,
)
test_model.eval()

# System prompt
SYSTEM_PROMPT = "You are a navigation assistant helping the user locate a target object inside a building.You will receive a sequence of frames describing visible objects. Each object includes:  - the floor,  - the relative position to the viewer,  - the distance from the viewer,  - and the room it belongs to.The frames appear in chronological order along the user's path from the starting point toward the target.Before starting the walk description, consider an initial turn direction if provided.Your task is to write a human-sounding description of the path.  Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).Mention at least one and at most two objects per room, choosing only the most informative for navigation.  If the path includes stairs, simply write: \"go up/down the stairs to reach the <room_name>\", without describing objects on the stairs.If you see the target location or object, mention it immediately and stop referencing any further objects.Only refer to objects that appear in the observations. Never invent or embellish details.Use the object IDs when referencing them (e.g., 'chair_5').You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. Imagine you are moving from the starting room to the target location, and provide clear path instructions."

# Test sample
TEST_PROMPT = "User question: How do I get to the upper bedroom? Observations: Initially, turn left. In frame-000000 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], picture_136 [(relative position: center-left), (room: living room)]. From previous frame, continue forward. In frame-000001 you are in living room on floor 0. You see stairs_170 [(relative position: lower-center), (room: living room)], couch_125 [(relative position: lower-right), (room: office)]. From previous frame, continue forward. In frame-000002 you are in office on floor 1. You see door_1 [(relative position: center-right), (room: upper bedroom)], wardrobe_27 [(relative position: lower-right), (room: upper bedroom)]. From previous frame, continue forward. In frame-000003 you are in upper bedroom on floor 1. You see bed_40 [(relative position: center-left), (room: upper bedroom)], armchair_72 [(relative position: lower-right), (room: upper bedroom)]. Rooms visited in order: living room (floor: 0), office (floor: 1), upper bedroom (floor: 1) The user is in living room (floor: 0) and the target is in upper bedroom (floor: 1)."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": TEST_PROMPT}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(test_model.device)
input_len = inputs['input_ids'].shape[1]

logger.info(f"Generating response for test sample...")

with torch.no_grad():
    outputs = test_model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

logger.info("\n" + "=" * 80)
logger.info("GENERATED RESPONSE:")
logger.info("=" * 80)
logger.info(response)
logger.info("=" * 80)
logger.info(f"\nTraining completed! Model saved to: {train_conf.output_dir}")

#######

########################################
#     INFERENCE TEST: BASE vs FINETUNED #
########################################

logger.info("\n" + "=" * 120)
logger.info(" " * 45 + "BASE vs FINETUNED COMPARISON")
logger.info("=" * 120)

# Load random samples from training data for comparison
import random

logger.info("\nLoading random samples from training data for comparison...")
with open("finetune_data.jsonl", 'r') as f:
    all_training_samples = [json.loads(line) for line in f]

NUM_TEST_SAMPLES = 15
random.seed(42)
comparison_samples = random.sample(all_training_samples, min(NUM_TEST_SAMPLES, len(all_training_samples)))

logger.info(f"Selected {len(comparison_samples)} random samples for testing\n")

# Prepare base model (without LoRA) for comparison
logger.info("Preparing base model for comparison...")
base_model_for_test = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    **model_kwargs
)
base_model_for_test.eval()

# Load finetuned model
logger.info("Loading finetuned model with LoRA adapter...")
finetuned_model = PeftModel.from_pretrained(
    model,
    train_conf.output_dir,
    torch_dtype=torch.bfloat16,
)
finetuned_model.eval()
logger.info("✅ Models loaded!\n")

########################################
#       INFERENCE FUNCTION             #
########################################

def generate_comparison_response(test_model, messages, model_name=""):
    """Generate response from a model"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(test_model.device)
    
    input_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response

########################################
#          RUN COMPARISON              #
########################################

results = []

for idx, sample in enumerate(comparison_samples, 1):
    messages = sample["messages"]
    
    # Extract system, user, and expected answer
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    expected = messages[2]["content"] if len(messages) > 2 else "N/A"
    
    logger.info(f"\n{'═' * 120}")
    logger.info(f"{'═' * 120}")
    logger.info(f" SAMPLE {idx}/{len(comparison_samples)} ".center(120))
    logger.info(f"{'═' * 120}")
    logger.info(f"{'═' * 120}")
    
    # Show user query (abbreviated)
    logger.info(f"\n{'─' * 120}")
    logger.info(f"❓ USER QUERY (first 150 chars):")
    logger.info(f"{'─' * 120}")
    logger.info(user_content[:150] + "...")
    logger.info(f"{'─' * 120}")
    
    # Generate with BASE model
    logger.info("⏳ Generating with BASE model...")
    base_response = generate_comparison_response(base_model_for_test, messages[:2], "BASE")
    
    # Generate with FINETUNED model
    logger.info("⏳ Generating with FINETUNED model...")
    finetuned_response = generate_comparison_response(finetuned_model, messages[:2], "FINETUNED")
    
    # Display results
    logger.info(f"\n{'━' * 120}")
    logger.info(f"🔵 BASE MODEL OUTPUT ({len(base_response.split())} words):")
    logger.info(f"{'━' * 120}")
    logger.info(base_response)
    
    logger.info(f"\n{'━' * 120}")
    logger.info(f"🟢 FINETUNED MODEL OUTPUT ({len(finetuned_response.split())} words):")
    logger.info(f"{'━' * 120}")
    logger.info(finetuned_response)
    
    logger.info(f"\n{'━' * 120}")
    logger.info(f"✅ EXPECTED GROUND TRUTH ({len(expected.split())} words):")
    logger.info(f"{'━' * 120}")
    logger.info(expected)
    logger.info(f"{'━' * 120}")
    
    # Store results
    results.append({
        'base_len': len(base_response.split()),
        'finetuned_len': len(finetuned_response.split()),
        'expected_len': len(expected.split()),
        'base_response': base_response,
        'finetuned_response': finetuned_response,
        'expected': expected,
    })

########################################
#          SUMMARY STATS               #
########################################

logger.info("\n" + "=" * 120)
logger.info("=" * 120)
logger.info(" 📊 SUMMARY STATISTICS ".center(120))
logger.info("=" * 120)
logger.info("=" * 120)

avg_base = sum(r['base_len'] for r in results) / len(results)
avg_finetuned = sum(r['finetuned_len'] for r in results) / len(results)
avg_expected = sum(r['expected_len'] for r in results) / len(results)

logger.info(f"\n📏 AVERAGE RESPONSE LENGTH:")
logger.info(f"   Base Model:      {avg_base:>6.1f} words")
logger.info(f"   Finetuned Model: {avg_finetuned:>6.1f} words")
logger.info(f"   Expected:        {avg_expected:>6.1f} words")

min_base = min(r['base_len'] for r in results)
max_base = max(r['base_len'] for r in results)
min_ft = min(r['finetuned_len'] for r in results)
max_ft = max(r['finetuned_len'] for r in results)
min_exp = min(r['expected_len'] for r in results)
max_exp = max(r['expected_len'] for r in results)

logger.info(f"\n📊 LENGTH RANGE:")
logger.info(f"   Base Model:      min={min_base:>3}, max={max_base:>3}")
logger.info(f"   Finetuned Model: min={min_ft:>3}, max={max_ft:>3}")
logger.info(f"   Expected:        min={min_exp:>3}, max={max_exp:>3}")

# Quality checks
over_120_base = sum(1 for r in results if r['base_len'] > 120)
over_120_ft = sum(1 for r in results if r['finetuned_len'] > 120)

logger.info(f"\n⚠️  RESPONSES EXCEEDING 120 WORDS:")
logger.info(f"   Base Model:      {over_120_base}/{len(results)}")
logger.info(f"   Finetuned Model: {over_120_ft}/{len(results)}")

if over_120_ft == 0:
    logger.info(f"\n✅ All finetuned responses are within 120 word limit!")
else:
    logger.info(f"\n⚠️  {over_120_ft} finetuned response(s) exceeded the limit")

# Length difference analysis
logger.info(f"\n📉 LENGTH IMPROVEMENT:")
avg_diff = avg_expected - avg_finetuned
logger.info(f"   Finetuned vs Expected: {avg_diff:+.1f} words difference")
if abs(avg_diff) < 10:
    logger.info(f"   ✅ Finetuned model matches expected length well!")
elif avg_diff > 0:
    logger.info(f"   ℹ️  Finetuned responses are slightly shorter than expected")
else:
    logger.info(f"   ℹ️  Finetuned responses are slightly longer than expected")

# Comparison analysis
logger.info(f"\n🎯 BASE vs FINETUNED COMPARISON:")
base_vs_expected = abs(avg_expected - avg_base)
ft_vs_expected = abs(avg_expected - avg_finetuned)
improvement = base_vs_expected - ft_vs_expected

logger.info(f"   Base model deviation from expected:      {base_vs_expected:.1f} words")
logger.info(f"   Finetuned model deviation from expected: {ft_vs_expected:.1f} words")
logger.info(f"   Improvement: {improvement:+.1f} words")

if improvement > 5:
    logger.info(f"   ✅ Finetuned model is significantly closer to expected length!")
elif improvement > 0:
    logger.info(f"   ✅ Finetuned model is closer to expected length")
else:
    logger.info(f"   ⚠️  Finetuned model is not closer to expected length")

logger.info("\n" + "=" * 120)
logger.info(" ✅ COMPARISON COMPLETE ".center(120))
logger.info("=" * 120)
logger.info(f"\nTraining completed! Model saved to: {train_conf.output_dir}")