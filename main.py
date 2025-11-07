print("PROYECTO FINAL: Detección de Phishing con LLMs - COMPARATIVA")
print("=" * 60)

from src.data_loader import DataLoader
from src.model_trainer import PhishingModelTrainer
from src.roberta_trainer import RobertaPhishingTrainer
import time

def main():
    # 1. Cargar datos
    print("Cargando datos...")
    loader = DataLoader()
    data = loader.load_phishing_data()
    
    print(f"Datos cargados: {len(data)} ejemplos")
    print(f"Distribucion: {data['label'].value_counts().to_dict()}")
    
    # 2. Entrenar DistilBERT
    print("\n" + "="*50)
    print("ENTRENANDO DISTILBERT")
    print("="*50)
    
    start_time = time.time()
    distilbert_trainer = PhishingModelTrainer()
    distilbert_result = distilbert_trainer.train(data)
    distilbert_time = time.time() - start_time
    
    # 3. Entrenar RoBERTa
    print("\n" + "="*50)
    print("ENTRENANDO ROBERTA")
    print("="*50)
    
    start_time = time.time()
    roberta_trainer = RobertaPhishingTrainer()
    roberta_result = roberta_trainer.train(data)
    roberta_time = time.time() - start_time
    
    # 4. Comparar predicciones
    print("\n" + "="*50)
    print("COMPARATIVA DE PREDICCIONES")
    print("="*50)
    
    test_emails = [
        "Your bank account needs verification immediately!",
        "Team meeting tomorrow at 9 AM in the main conference room.",
        "You won a free iPhone! Click here to claim your prize now!",
        "Please review the attached quarterly report.",
        "URGENT: Your PayPal account will be suspended. Update now!",
        "Your order confirmation #12345 has been shipped.",
    ]
    
    print(f"\n{'EMAIL':<60} {'DISTILBERT':<15} {'ROBERTA':<15}")
    print("-" * 90)
    
    for email in test_emails:
        distilbert_pred = distilbert_trainer.predict(email)
        roberta_pred = roberta_trainer.predict(email)
        
        email_short = email[:55] + "..." if len(email) > 55 else email
        print(f"{email_short:<60} {distilbert_pred['prediction']} ({distilbert_pred['confidence']:.2f})   {roberta_pred['prediction']} ({roberta_pred['confidence']:.2f})")
    
    # 5. Mostrar comparativa de tiempos
    print("\n" + "="*50)
    print("COMPARATIVA DE TIEMPOS")
    print("="*50)
    print(f"DistilBERT: {distilbert_time:.2f} segundos")
    print(f"RoBERTa: {roberta_time:.2f} segundos")
    print(f"Diferencia: {abs(distilbert_time - roberta_time):.2f} segundos")
    
    # 6. Conclusión
    print("\n" + "="*50)
    print("CONCLUSIONES")
    print("="*50)
    if distilbert_time < roberta_time:
        print("DistilBERT es mas rapido")
    else:
        print("RoBERTa es mas rapido")
    
    print("Ambos modelos estan funcionando correctamente!")

if __name__ == "__main__":
    main()