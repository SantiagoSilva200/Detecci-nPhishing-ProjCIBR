import pandas as pd
import requests
import io
from datasets import load_dataset

class DataLoader:
    def __init__(self):
        self.dataset = None
        
    def load_phishing_data(self):
        """Carga multiples datasets de phishing"""
        print("Cargando multiples datasets de phishing...")
        
        all_datasets = []
        
        # 1. Dataset UCI Phishing Websites
        print("1. Cargando dataset UCI...")
        uci_data = self._load_uci_phishing()
        if uci_data is not None:
            all_datasets.append(uci_data)
            print(f"   UCI cargado: {len(uci_data)} ejemplos")
        
        # 2. Dataset de Emails Phishing (de Hugging Face)
        print("2. Cargando dataset de emails...")
        email_data = self._load_email_dataset()
        if email_data is not None:
            all_datasets.append(email_data)
            print(f"   Emails cargados: {len(email_data)} ejemplos")
        
        # 3. Dataset de respaldo (sintético)
        print("3. Cargando dataset sintetico...")
        synthetic_data = self._create_synthetic_dataset()
        all_datasets.append(synthetic_data)
        print(f"   Sintetico cargado: {len(synthetic_data)} ejemplos")
        
        # Combinar todos los datasets
        if len(all_datasets) > 1:
            combined_data = pd.concat(all_datasets, ignore_index=True)
        else:
            combined_data = all_datasets[0]
        
        # Mezclar los datos
        combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"DATASET COMBINADO: {len(combined_data)} ejemplos totales")
        print(f"Distribucion final: {combined_data['label'].value_counts().to_dict()}")
        
        return combined_data
    
    def _load_uci_phishing(self):
        """Carga dataset UCI de phishing websites"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
            response = requests.get(url)
            
            if response.status_code == 200:
                from scipy.io import arff
                data, meta = arff.loadarff(io.StringIO(response.text))
                df = pd.DataFrame(data)
                
                # Convertir bytes a strings
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].str.decode('utf-8')
                
                # Crear textos mas variados basados en las caracteristicas
                phishing_texts = []
                legit_texts = []
                
                for i in range(min(500, len(df))):
                    if df.iloc[i]['Result'] == '-1':  # Phishing
                        phishing_texts.extend([
                            f"URGENT SECURITY ALERT: Suspicious activity detected on account {i}. Verify immediately: http://secure-login-{i}.com",
                            f"Congratulations! You are the lucky winner of ${i*100}. Claim your reward: http://prize-claim-{i}.net",
                            f"Your financial account #{i} has been temporarily locked. Unlock now: http://account-recovery-{i}.com",
                            f"IMPORTANT: Package delivery #{i} failed. Confirm your shipping address: http://tracking-update-{i}.com"
                        ])
                    else:  # Legitimo
                        legit_texts.extend([
                            f"Meeting confirmation for project reference #{i}. Scheduled for tomorrow at 10 AM.",
                            f"Your invoice #INV-2024-{i} has been processed and is ready for payment.",
                            f"Order update: Your items for order #{i} have been shipped successfully.",
                            f"Account notification: Subscription service #{i} has been renewed automatically."
                        ])
                
                # Balancear
                min_count = min(len(phishing_texts), len(legit_texts), 100)  # Maximo 100 de cada
                texts = phishing_texts[:min_count] + legit_texts[:min_count]
                labels = [1] * min_count + [0] * min_count
                
                return pd.DataFrame({'text': texts, 'label': labels})
                
        except Exception as e:
            print(f"   Error UCI: {e}")
            return None
    
    def _load_email_dataset(self):
        """Carga dataset real de emails phishing"""
        try:
            # Intentar cargar dataset de emails phishing de Hugging Face
            print("   Intentando cargar dataset de emails reales...")
            
            # Opcion 1: Dataset de phishing emails
            try:
                dataset = load_dataset("shreydan/phishing-email", split="train")
                df = pd.DataFrame({
                    'text': dataset['text'],
                    'label': dataset['label']
                })
                print(f"   Dataset de emails real cargado: {len(df)} ejemplos")
                return df
            except:
                pass
            
            # Opcion 2: Dataset alternativo
            try:
                dataset = load_dataset("california-institute/Phishing-Email", split="train")
                df = pd.DataFrame({
                    'text': dataset['text'],
                    'label': [1 if 'phishing' in str(t).lower() else 0 for t in dataset['label']]
                })
                print(f"   Dataset alternativo cargado: {len(df)} ejemplos")
                return df
            except:
                pass
            
            # Opcion 3: Crear dataset de emails manual
            print("   Creando dataset de emails manual...")
            return self._create_email_manual_dataset()
            
        except Exception as e:
            print(f"   Error emails: {e}")
            return None
    
    def _create_email_manual_dataset(self):
        """Crea dataset manual de emails phishing/legitimos"""
        phishing_emails = [
            # Urgency & Security
            "URGENT: Your bank account verification required immediately. Click here: http://bank-security-update.com",
            "SECURITY ALERT: Unusual login activity detected. Secure your account now: http://account-protection.net",
            "IMMEDIATE ACTION: Your password has been compromised. Reset here: http://password-reset-secure.com",
            
            # Prizes & Money
            "CONGRATULATIONS! You won $10,000 Walmart gift card. Claim: http://walmart-rewards-center.com",
            "You are our 1,000,000th visitor! Claim your $500 Amazon gift card: http://amazon-gift-promo.com",
            "INHERITANCE NOTICE: $150,000 unclaimed funds waiting. Contact: http://inheritance-claims.org",
        ]
        
        legit_emails = [
            # Work & Meetings
            "Meeting reminder: Project review scheduled for tomorrow at 2 PM in conference room B",
            "Team lunch this Friday at 12:30 PM at the downtown restaurant",
            "Weekly status meeting agenda attached for your review",
            
            # Invoices & Payments
            "Your invoice #INV-2024-789 is ready for payment. Due date: 30 days",
            "Payment confirmation for order #ORD-456. Thank you for your business",
            "Monthly subscription invoice attached for your records",
        ]
        
        # Balancear
        texts = phishing_emails + legit_emails
        labels = [1] * len(phishing_emails) + [0] * len(legit_emails)
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    def _create_synthetic_dataset(self):
        """Dataset sintetico de respaldo"""
        sample_emails = [
            # Phishing
            "Congratulations! You won $1000 prize. Click here to claim your reward now!",
            "Your account has been compromised. Verify your identity immediately.",
            "Urgent! Your PayPal account will be suspended. Update your information.",
            
            # Legitimate
            "Meeting scheduled for tomorrow at 10 AM in conference room B.",
            "Your invoice #INV-2023-456 is ready for payment.",
            "Hi team, please find attached the quarterly report for review.",
        ]
        
        labels = [1, 1, 1, 0, 0, 0]
        
        return pd.DataFrame({'text': sample_emails, 'label': labels})

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_phishing_data()
    print(f"DATASET FINAL LISTO!")
    print(f"Total ejemplos: {len(data)}")
    print(f"Distribucion: {data['label'].value_counts()}")