import polars as pl
import os
import re
from datetime import date, timedelta

# Definir el directorio base donde se encuentran las carpetas
base_dir = 'results'

# Inicializar una lista para almacenar los DataFrames
df_list = []

# Definir el patrón fijo del nombre de las carpetas
pattern = r'informer_single_run_lgbm_total_\d{8}_ftS_sl28_ll7_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_'

# Recorrer todas las carpetas en el directorio base
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Verificar si el nombre de la carpeta coincide con el patrón deseado
    if re.match(pattern, folder_name):
        
        # Definir la ruta del archivo CSV
        csv_path = os.path.join(folder_path, 'real_prediction.csv')
        
        # Verificar si el archivo existe en la carpeta
        if os.path.isfile(csv_path):
            # Leer el archivo CSV y agregarlo a la lista de DataFrames
            df = pl.read_csv(csv_path)
            df_list.append(df)

# Concatenar todos los DataFrames en uno solo
df_final = pl.concat(df_list)

# Convertir la columna 'date' a formato de fecha si no lo está
df_final = df_final.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

# Ordenar el DataFrame por la columna 'date'
df_final = df_final.sort("date")

# Guardar el DataFrame resultante en un archivo CSV
output_path = 'resultados_concatenados.csv'
df_final.write_csv(output_path)

# Mostrar el DataFrame resultante
print(df_final)
