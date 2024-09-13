import os
import subprocess

# Configuración
input_dir = "data/logtel/"
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
# print(len(csv_files))

files_to_remove = ['cantidad_entregas_total_20211202.csv', 'cantidad_entregas_total_20220728.csv', 'cantidad_entregas_total_20220224.csv', 'cantidad_entregas_total_20211118.csv', 'cantidad_entregas_total_20210527.csv', 'cantidad_entregas_total_20191010.csv', 'cantidad_entregas_total_20191205.csv', 'cantidad_entregas_total_20210429.csv', 'cantidad_entregas_total_20200409.csv', 'cantidad_entregas_total_20200430.csv', 'cantidad_entregas_total_20200604.csv', 'cantidad_entregas_total_20220331.csv', 'cantidad_entregas_total_20210909.csv', 'cantidad_entregas_total_20191017.csv', 'cantidad_entregas_total_20210218.csv', 'cantidad_entregas_total_20201022.csv', 'cantidad_entregas_total_20211021.csv', 'cantidad_entregas_total_20230112.csv', 'cantidad_entregas_total_20220106.csv', 'cantidad_entregas_total_20230223.csv', 'cantidad_entregas_total_20200102.csv', 'cantidad_entregas_total_20220407.csv', 'cantidad_entregas_total_20231207.csv', 'cantidad_entregas_total_20220804.csv', 'cantidad_entregas_total_20240606.csv', 'cantidad_entregas_total_20200227.csv', 'cantidad_entregas_total_20230831.csv', 'cantidad_entregas_total_20201001.csv', 'cantidad_entregas_total_20240111.csv', 'cantidad_entregas_total_20231123.csv', 'cantidad_entregas_total_20240627.csv', 'cantidad_entregas_total_20200528.csv', 'cantidad_entregas_total_20220908.csv', 'cantidad_entregas_total_20240411.csv', 'cantidad_entregas_total_20211111.csv', 'cantidad_entregas_total_20200213.csv', 'cantidad_entregas_total_20201231.csv', 'cantidad_entregas_total_20230406.csv', 'cantidad_entregas_total_20240425.csv', 'cantidad_entregas_total_20211223.csv', 'cantidad_entregas_total_20191003.csv', 'cantidad_entregas_total_20220210.csv', 'cantidad_entregas_total_20200109.csv', 'cantidad_entregas_total_20221201.csv', 'cantidad_entregas_total_20230601.csv', 'cantidad_entregas_total_20240418.csv', 'cantidad_entregas_total_20240125.csv', 'cantidad_entregas_total_20220707.csv', 'cantidad_entregas_total_20230615.csv', 'cantidad_entregas_total_20220526.csv', 'cantidad_entregas_total_20230525.csv', 'cantidad_entregas_total_20221027.csv', 'cantidad_entregas_total_20200206.csv', 'cantidad_entregas_total_20240314.csv', 'cantidad_entregas_total_20200910.csv', 'cantidad_entregas_total_20210826.csv', 'cantidad_entregas_total_20210701.csv', 'cantidad_entregas_total_20201008.csv', 'cantidad_entregas_total_20211028.csv', 'cantidad_entregas_total_20200917.csv', 'cantidad_entregas_total_20221110.csv', 'cantidad_entregas_total_20230921.csv', 'cantidad_entregas_total_20220609.csv', 'cantidad_entregas_total_20220825.csv', 'cantidad_entregas_total_20211216.csv', 'cantidad_entregas_total_20200827.csv', 'cantidad_entregas_total_20210603.csv', 'cantidad_entregas_total_20210506.csv', 'cantidad_entregas_total_20240620.csv', 'cantidad_entregas_total_20210422.csv', 'cantidad_entregas_total_20240321.csv', 'cantidad_entregas_total_20230413.csv', 'cantidad_entregas_total_20220113.csv', 'cantidad_entregas_total_20200507.csv', 'cantidad_entregas_total_20240523.csv', 'cantidad_entregas_total_20220721.csv', 'cantidad_entregas_total_20191128.csv', 'cantidad_entregas_total_20220811.csv', 'cantidad_entregas_total_20231109.csv', 'cantidad_entregas_total_20210408.csv', 'cantidad_entregas_total_20230202.csv', 'cantidad_entregas_total_20211007.csv', 'cantidad_entregas_total_20200716.csv', 'cantidad_entregas_total_20210513.csv', 'cantidad_entregas_total_20210902.csv', 'cantidad_entregas_total_20220127.csv', 'cantidad_entregas_total_20230302.csv', 'cantidad_entregas_total_20240725.csv', 'cantidad_entregas_total_20230824.csv', 'cantidad_entregas_total_20230914.csv', 'cantidad_entregas_total_20200130.csv', 'cantidad_entregas_total_20230316.csv', 'cantidad_entregas_total_20230803.csv', 'cantidad_entregas_total_20240711.csv', 'cantidad_entregas_total_20200326.csv', 'cantidad_entregas_total_20210812.csv', 'cantidad_entregas_total_20210211.csv', 'cantidad_entregas_total_20231221.csv']

# Filtrar la lista de archivos eliminando los que están en files_to_remove
csv_files = [f for f in csv_files if f in files_to_remove]


#print(len(csv_files))
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
