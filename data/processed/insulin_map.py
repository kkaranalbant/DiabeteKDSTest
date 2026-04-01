# import polars as pl
#
# # Eğitimde kullandığın dosya yolunu buraya yaz
# # (v6 modelini eğittiğin "engineered" dosya olmalı)
# FILE_PATH = r"D:\Diabete\data\processed\master_train_v6_engineered.parquet"
#
# print("📂 Veri seti taranıyor...")
#
# # Sadece ilgili sütunları oku
# try:
#     df = pl.read_parquet(FILE_PATH, columns=['insulin_type_bolus', 'insulin_type_encoded'])
#
#     # Benzersiz eşleşmeleri bul ve sırala
#     mapping_df = (
#         df.unique(subset=['insulin_type_encoded'])
#         .sort('insulin_type_encoded')
#         .select(['insulin_type_encoded', 'insulin_type_bolus'])
#     )
#
#     print("\n✅ KESİN İNSÜLİN HARİTASI (Python Dictionary Formatında):")
#     print("-" * 50)
#
#     # Dictionary formatında yazdır (Kopyalayıp app.py'ye yapıştırmak için)
#     print("INSULIN_MAP = {")
#     for row in mapping_df.iter_rows(named=True):
#         # Eğer 'insulin_type_bolus' bir liste veya string ise temizle
#         name = str(row['insulin_type_bolus'])
#         code = row['insulin_type_encoded']
#         print(f'    "{name}": {code},')
#     print("}")
#     print("-" * 50)
#
# except Exception as e:
#     print(f"❌ Hata: {e}")
#     print("Dosya yolunu veya sütun isimlerini kontrol et.")

import pyarrow.parquet as pq

parquet_file = pq.ParquetFile(r"D:\Diabete\data\raw\metabonet_public_train.parquet")

columns_list = parquet_file.schema.names

print(columns_list)

import pandas as pd

df = pd.read_parquet(r"D:\Diabete\data\raw\metabonet_public_train.parquet")

numeric_columns = df.select_dtypes(include="number").columns.tolist()

print(numeric_columns)
