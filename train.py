import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class ImageHtmlDataset(Dataset):
    def __init__(self, dataset_path, clip_model, clip_processor, tokenizer):
        self.dataset_path = dataset_path
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer

        self.image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(image_path))[0]
        html_path = os.path.join(self.dataset_path, f"{basename}.html")

        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_embeds = self.clip_model.get_image_features(**inputs).squeeze(0)  # [512]

        with open(html_path, "r") as f:
            html = f.read()
        
        html_tokens = self.tokenizer(
            html,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length",
            add_special_tokens=True
        )

        input_ids = html_tokens['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = html_tokens['attention_mask'].squeeze(0)  # [seq_len]

        return image_embeds, input_ids, attention_mask


# Collate function for batching
def collate_fn(batch):
    image_embeds_batch = torch.stack([item[0] for item in batch])  # [B, 512]
    input_ids_batch = [item[1] for item in batch]
    attention_mask_batch = [item[2] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)  # [B, max_seq_len]
    attention_mask_padded = pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)  # [B, max_seq_len]
    
    return image_embeds_batch, input_ids_padded, attention_mask_padded


# Prefix tuning model
class PrefixTuningModel(nn.Module):
    def __init__(self, codegen_model, prefix_length=50, embed_dim=2048):
        super().__init__()
        self.codegen = codegen_model
        self.prefix_length = prefix_length
        self.embed_dim = embed_dim

        for param in self.codegen.parameters():
            param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, prefix_length * embed_dim)
        )
    
    def forward(self, image_embeds, html_token_ids, attention_mask):
        """
        image_embeds: [batch_size, 512]
        html_token_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        """
        batch_size = image_embeds.shape[0]

        prefix_embeds = self.projector(image_embeds)  # [B, prefix_len * embed_dim]
        prefix_embeds = prefix_embeds.view(batch_size, self.prefix_length, self.embed_dim)  # [B, prefix_len, embed_dim]

        html_embeds = self.codegen.transformer.wte(html_token_ids)  # [B, seq_len, embed_dim]

        inputs_embeds = torch.cat((prefix_embeds, html_embeds), dim=1)  # [B, prefix_len + seq_len, embed_dim]
        
        # Create attention mask for the combined sequence
        prefix_attention_mask = torch.ones((batch_size, self.prefix_length), device=attention_mask.device)
        combined_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # Create position IDs for the combined sequence
        position_ids = torch.arange(0, self.prefix_length + html_token_ids.size(1), dtype=torch.long, device=html_token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        outputs = self.codegen(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            position_ids=position_ids
        )

        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_dataset_path = "/root/image_to_html/kaggle_dataset/train"
    test_dataset_path = "/root/image_to_html/kaggle_dataset/test"

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    tokenizer.pad_token = tokenizer.eos_token
    codegen = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(device)

    train_dataset = ImageHtmlDataset(train_dataset_path, clip_model, clip_processor, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    prefix_length = 50
    embed_dim = codegen.config.hidden_size
    prefix_tuning_model = PrefixTuningModel(codegen, prefix_length=prefix_length, embed_dim=embed_dim).to(device)
    prefix_tuning_model.train()

    optimizer = torch.optim.Adam(prefix_tuning_model.projector.parameters(), lr=1e-4)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1): 
        for image_embeds, html_token_ids, attention_mask in tqdm(train_loader):
            image_embeds = image_embeds.to(device)
            html_token_ids = html_token_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            outputs = prefix_tuning_model(image_embeds, html_token_ids, attention_mask)
            logits = outputs.logits  

            batch_size, seq_len = html_token_ids.size()

            labels = torch.cat([
                torch.full((batch_size, prefix_length), -100, device=device),
                html_token_ids
            ], dim=1)

            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    print("Training complete.")





