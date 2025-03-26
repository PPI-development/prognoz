import os
import pandas as pd

def process_excel_files():
    directory = os.getcwd()
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.xls')]
    
    for file in files:
        file_path = os.path.join(directory, file)
        print(f'Обрабатываю файл: {file_path}')
        
        try:
            df = pd.read_excel(file_path, skiprows=2, header=None)
            
            df.columns = [
                'datetime', 'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm',
                'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm'
            ]
            
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            df['year'] = df['datetime'].dt.year
            
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
            
            numeric_columns = [
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm',
                'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm'
            ]
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df_mean_by_year = df.groupby('year').mean(numeric_only=True)
            
            output_file = os.path.join(directory, f"{os.path.splitext(file)[0]}_mean_by_year.csv")
            df_mean_by_year.to_csv(output_file)
            print(f'Файл сохранен: {output_file}')
        
        except Exception as e:
            print(f'Ошибка при обработке файла {file_path}: {e}')

if __name__ == "__main__":
    process_excel_files()