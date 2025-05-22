import torch
from transformers import AutoTokenizer, AutoProcessor, CLIPModel, AutoModelForCausalLM
from PIL import Image
import os
from train import PrefixTuningModel

# ----------------------------
# Configs
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models & Paths
CODEGEN_MODEL_NAME = "Salesforce/codegen-350M-mono"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
PREFIX_TUNING_MODEL_PATH = "prefix_tuning_model_2.pt"
INPUT_IMAGE_PATH = "/root/image_to_html/processed_files/test/0CE73E18-575A-4A70-9E40-F000B250344F.png" 

# Parameters
PREFIX_LENGTH = 50
MAX_LENGTH = 2048

# ----------------------------
# Load models
# ----------------------------
print("Loading models...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(CODEGEN_MODEL_NAME)

# CLIP
clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(device)

# CodeGen
codegen = AutoModelForCausalLM.from_pretrained(CODEGEN_MODEL_NAME).eval().to(device)
embed_dim = codegen.transformer.wte.weight.shape[1]


prefix_tuning_model = PrefixTuningModel(codegen, embed_dim=embed_dim, prefix_length=PREFIX_LENGTH).to(device)
prefix_tuning_model.load_state_dict(torch.load(PREFIX_TUNING_MODEL_PATH, map_location=device))
prefix_tuning_model.eval()

# ----------------------------
# Load and process image
# ----------------------------
print("Processing image...")

image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
inputs = clip_processor(images=image, return_tensors="pt").to(device)

# Get image embeddings from CLIP
with torch.no_grad():
    image_embeds = clip_model.get_image_features(**inputs).squeeze(0)  # [512]
    image_embeds = image_embeds.unsqueeze(0)  # Add batch dimension [1, 512]

# After processing image...

# Start with a meaningful prompt
prompt = "<html>"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids).to(device)

# Generation parameters
max_new_tokens = 500  # Limit new tokens to generate
temperature = 0.7    # Add some randomness
generated_ids = input_ids

print("Generating HTML...")
for i in range(max_new_tokens):
    with torch.no_grad():
        outputs = prefix_tuning_model(image_embeds, generated_ids, attention_mask)
        next_token_logits = outputs.logits[:, -1, :] / temperature
        
        # Apply softmax to get probabilities
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.ones_like(generated_ids)
        
        # Print progress every 50 tokens
        if i % 50 == 0:
            current_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated {i} tokens...")
            print(f"Current output: {current_output[-100:]}...")  # Show last 100 chars
        
        # Stop if we generate an end token or closing HTML tag
        if next_token.item() == tokenizer.eos_token_id or "</html>" in tokenizer.decode(generated_ids[0]):
            print("Generation complete!")
            break

html_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nFinal Generated HTML:\n")
print(html_output)


# ----------------------------
# Output
# ----------------------------
print("\nGenerated HTML:\n")
print(html_output)
