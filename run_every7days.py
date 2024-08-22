import os
import subprocess

# Configuración
input_dir = "data/logtel/"
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    # Extraer la fecha del nombre del archivo
    data_date = csv_file.split('_')[-1].replace('.csv', '')
    
    # Establecer la variable de entorno DATA_DATE
    os.environ['DATA_DATE'] = data_date
    print(data_date)
    print(csv_file)
    
    # Ejecutar el script run.py
    command = ["python", "run.py"]
    subprocess.run(command)
