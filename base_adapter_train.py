import torch
import torchaudio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from evaluate import load

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

class CzechSpeechDataset(Dataset):
    def __init__(self, dataset_split, processor=processor, sampling_rate=16000):

        self.dataset = load_dataset("shunyalabs/czech-speech-dataset", split=dataset_split)
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]
        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]
        transcript = item["transcript"]

        if sample_rate != self.sampling_rate:
            waveform = torch.from_numpy(audio_array).float()
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.sampling_rate
            )(waveform)
            audio_array = waveform.numpy()

        inputs = self.processor(
            audio_array, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        )
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
            "transcript": transcript
        }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feat["input_features"]} for feat in features]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=True,
            return_tensors="pt"
        )

        labels = [{"input_ids": feat["labels"]} for feat in features]
        labels_batch = self.processor.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (
            labels.size(1) > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

wer_metric = load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    warmup_steps=0,
    num_train_epochs=3,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_first_step=True,
    logging_nan_inf_filter=False,
    eval_steps=500,
    fp16=True,
    per_device_eval_batch_size=4,
    generation_max_length=128,
    logging_steps=1,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True,  
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", 
    device_map="auto"
)

model.generation_config.language = "czech"
model.generation_config.task = "transcribe"

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = CzechSpeechDataset("train")
eval_dataset = CzechSpeechDataset("validation")

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

if len(train_dataset) < 1000:
    training_args.learning_rate = 5e-4
    training_args.num_train_epochs = 5
    print("Small dataset detected: adjusted learning rate to 5e-4 and epochs to 5")

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

model.config.use_cache = False

trainer.train()

trainer.save_model("whisper-large-czech-lora")
print("Training completed and model saved!")