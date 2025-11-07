from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

class RobertaPhishingTrainer:
    def __init__(self, model_name="roberta-base"):
        print("Cargando modelo RoBERTa y tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trainer = None
        print("Modelo RoBERTa cargado correctamente!")
        
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def prepare_data(self, df):
        """Prepara los datos para entrenamiento"""
        dataset = Dataset.from_pandas(df)
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        return train_test_split['train'], train_test_split['test']
    
    def train(self, df):
        """Entrena el modelo RoBERTa"""
        print("Preparando datos para RoBERTa...")
        
        train_data, eval_data = self.prepare_data(df)
        
        print(f"Datos de entrenamiento: {len(train_data)} ejemplos")
        print(f"Datos de evaluacion: {len(eval_data)} ejemplos")
        
        # Tokenizar datos
        print("Tokenizando datos para RoBERTa...")
        tokenized_train = train_data.map(self.tokenize_function, batched=True)
        tokenized_eval = eval_data.map(self.tokenize_function, batched=True)
        
        # Configurar entrenamiento para RoBERTa
        training_args = TrainingArguments(
            output_dir="./roberta_results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./roberta_logs",
            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",
        )
        
        print("Configurando entrenamiento de RoBERTa...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
        )
        
        print("Iniciando entrenamiento de RoBERTa...")
        training_result = self.trainer.train()
        
        print("Entrenamiento de RoBERTa completado!")
        
        # Evaluar
        if len(tokenized_eval) > 0:
            print("Evaluando modelo RoBERTa...")
            eval_results = self.trainer.evaluate(tokenized_eval)
            print(f"Resultados de RoBERTa:")
            for key, value in eval_results.items():
                print(f"   {key}: {value:.4f}")
        
        return training_result
    
    def predict(self, text):
        """Hace prediccion en texto nuevo con RoBERTa"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax().item()
        confidence = predictions.max().item()
        
        return {
            'prediction': 'PHISHING' if predicted_class == 1 else 'LEGITIMO',
            'confidence': confidence,
            'class': predicted_class
        }

if __name__ == "__main__":
    print("Probando RoBERTa trainer...")
    trainer = RobertaPhishingTrainer()
    print("RoBERTa cargado correctamente!")