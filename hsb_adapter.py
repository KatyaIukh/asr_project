import os
import torch
import torchaudio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import json
from evaluate import load
import matplotlib.pyplot as plt
import numpy as np

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
wer_metric = load("wer")

def compute_metrics(pred):
    """Вычисление WER и точности распознавания"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    word_accuracy = 1 - wer
    
    return {"wer": wer, "word_accuracy": word_accuracy}

class AudioTextDataset(Dataset):
    """Датасет для аудиофайлов и транскрипций"""
    def __init__(self, json_path, processor=processor, sampling_rate=16000):
        with open(json_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_path"]
        transcript = item["transcript"]

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)(waveform)
        waveform = waveform.squeeze(0).numpy()

        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
        }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feat["input_features"]} for feat in features]
        batch = self.processor.feature_extractor.pad(input_features, padding=True, return_tensors="pt")

        labels = [{"input_ids": feat["labels"]} for feat in features]
        labels_batch = self.processor.tokenizer.pad(labels, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels.size(1) > 1 and 
            (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("Загрузка модели...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", device_map="auto")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="cs", task="transcribe")
model.generation_config.forced_decoder_ids = forced_decoder_ids

model = prepare_model_for_kbit_training(model)

# Конфигурация LoRA адаптера
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-small-sorbian-lora_nocz",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=0,
    num_train_epochs=3,
    evaluation_strategy="steps",
    logging_strategy="steps",
    eval_steps=500,
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True,
    generation_config=model.generation_config,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=AudioTextDataset(json_path="sorbian_train.json"),
    eval_dataset=AudioTextDataset(json_path="sorbian_val.json"),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

model.config.use_cache = False

print("Запуск обучения адаптера...")
trainer.train()

# 1. Средний WER на валидационном наборе
val_results = trainer.evaluate(AudioTextDataset(json_path="sorbian_val.json"))
print(f"Средний WER: {val_results['eval_wer']:.4f}")

# 2. WER на каждом примере + гистограмма
import matplotlib.pyplot as plt
import numpy as np

def evaluate_per_sample(model, dataset, processor):
    model.eval()
    wers = []
    predictions_list = []
    references_list = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                example = dataset[i]
                input_features = example["input_features"].unsqueeze(0).to(model.device)
                
                generated_ids = model.generate(
                    input_features,
                    max_length=128,
                    num_beams=1,
                    do_sample=False
                )
                
                prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                reference = processor.batch_decode(example["labels"].unsqueeze(0), skip_special_tokens=True)[0]
                
                wer = wer_metric.compute(predictions=[prediction], references=[reference])
                wers.append(wer)
                predictions_list.append(prediction)
                references_list.append(reference)
                    
            except Exception as e:
                print(f"Ошибка на примере {i}: {e}")
                wers.append(1.0)
    
    return wers, predictions_list, references_list

print("Оценка WER для каждого примера...")
val_dataset = AudioTextDataset(json_path="sorbian_val.json")
wers, predictions, references = evaluate_per_sample(model, val_dataset, processor)

# 3. Гистограмма распределения WER
plt.figure(figsize=(8, 5))
plt.hist(wers, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(np.mean(wers), color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {np.mean(wers):.3f}')
plt.xlabel('WER')
plt.ylabel('Количество примеров')
plt.title('Распределение WER по примерам')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wer_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Несколько примеров
print("\nПримеры предсказаний:")
print("-" * 50)
for i in range(min(5, len(wers))):
    print(f"Пример {i+1}:")
    print(f"  WER: {wers[i]:.3f}")
    print(f"  Эталон: {references[i]}")
    print(f"  Предсказание: {predictions[i]}")
    print()

# Сохраняем модель
trainer.save_model("whisper-small-sorbian-lora")
print("Обучение и оценка завершены!")


