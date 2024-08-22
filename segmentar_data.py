import polars as pl
from datetime import datetime, timedelta
import os

# Configuración
input_csv = "data/logtel/cantidad_entregas_total_20240820.csv"
output_dir = "data/logtel/"

# Leer el CSV completo con Polars
df = pl.read_csv(input_csv)
df = df.with_columns(pl.col('date').str.strptime(pl.Date, '%Y-%m-%d'))

# Definir la ventana inicial
start_date = datetime(2019, 9, 5)
end_date = start_date + timedelta(days=28)

#Loop para hacer las predicciones por ventanas de 7 días
while end_date <= datetime(2024,8,18):
    # Filtrar los datos de la ventana actual
    df_window = df.filter((pl.col('date') < end_date))
    
    # Guardar el archivo CSV temporal con la ventana de datos
    end_date_str = end_date.strftime('%Y%m%d')
    file_name = f"cantidad_entregas_total_{end_date_str}.csv"
    output_file = os.path.join(output_dir, file_name)

    # Guardar el CSV segmentado
    df_window.write_csv(output_file)
    print(f"Saved {output_file}")

    # Avanzar la ventana de tiempo
    end_date += timedelta(days=7)
