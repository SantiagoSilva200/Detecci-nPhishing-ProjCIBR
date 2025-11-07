from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

class PhishingModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        print("Cargando modelo y tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trainer = None
        print("Modelo DistilBERT cargado correctamente!")
        
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
        """Entrena el modelo con los datos"""
        print("Preparando datos para entrenamiento...")
        
        train_data, eval_data = self.prepare_data(df)
        
        print(f"Datos de entrenamiento: {len(train_data)} ejemplos")
        print(f"Datos de evaluacion: {len(eval_data)} ejemplos")
        
        # Tokenizar datos
        print("Tokenizando datos...")
        tokenized_train = train_data.map(self.tokenize_function, batched=True)
        tokenized_eval = eval_data.map(self.tokenize_function, batched=True)
        
        # Configurar entrenamiento
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
        )
        
        print("Configurando entrenamiento...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
        )
        
        print("Iniciando entrenamiento...")
        training_result = self.trainer.train()
        
        print("Entrenamiento completado!")
        
        # Evaluar manualmente
        if len(tokenized_eval) > 0:
            print("Evaluando modelo con datos de prueba...")
            eval_results = self.trainer.evaluate(tokenized_eval)
            print(f"Resultados de evaluacion:")
            for key, value in eval_results.items():
                print(f"   {key}: {value:.4f}")
        
        return training_result
    
    def predict(self, text):
        """Hace prediccion en texto nuevo"""
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
    print("Probando el trainer...")
    trainer = PhishingModelTrainer()
    print("Modelo cargado correctamente!")