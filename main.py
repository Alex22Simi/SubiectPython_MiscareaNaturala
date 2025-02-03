import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#cerinta A1
# Citim fișierele CSV
df_miscare_naturala = pd.read_csv('MiscareaNaturala.csv')


#  Selectăm coloanele necesare pentru cerința 1
df_miscare_naturala = df_miscare_naturala[['Three_Letter_Country_Code', 'Country_Name', 'RS']]
df_miscare_naturala = df_miscare_naturala[df_miscare_naturala['RS'] > 0]  # Filtrăm doar RS pozitiv
df_miscare_naturala = df_miscare_naturala.sort_values(by='RS', ascending=False)  # Sortare descrescătoare

# Salvăm rezultatul în Cerinta_1.csv
df_miscare_naturala.to_csv('Cerinta_1.csv', index=False)


#cerinta A.2

df_natural = pd.read_csv('MiscareaNaturala.csv')
df_coduri = pd.read_csv('CoduriTariExtins.csv')

#selectare coloane care trebuiesc
df_natural = df_natural[['Three_Letter_Country_Code','FR','IM','LE','LEF','LEM','MMR','RS']]
df_coduri = df_coduri[['Three_Letter_Country_Code','Continent_Name']]

#convertire coloane numerice
df_numerics = ['FR','IM','LE','LEF','LEM','MMR','RS']
df_natural[df_numerics] = df_natural[df_numerics].apply(pd.to_numeric,errors='coerce')

#facem merge
df_merge = pd.merge(df_natural, df_coduri, on='Three_Letter_Country_Code', how='left')

#calculare medie indici pentru fiecare continent
df_medie_indici = df_merge.groupby('Continent_Name',)[df_numerics].mean()

#calculare medie globara spor
mean_global_rs = df_medie_indici['RS'].mean()

#selctare continente rs  > media globala
df_filtered = df_medie_indici[df_medie_indici['RS']> mean_global_rs]

df_filtered = df_filtered.reset_index()

df_filtered.to_csv('Cerinta2.csv',index=False)

#B.1.
#sa se efectueze analiza in compo principale standardizata
#si sa se furnizeze rezultatele:
#variantele comp principale
#variantele afisate la consola

# Citire date
df = pd.read_csv('MiscareaNaturala.csv')

# Selectăm coloanele numerice
cols_numeric = ['FR', 'IM', 'LE', 'LEF', 'LEM', 'MMR', 'RS']
data = df[cols_numeric]

# Standardizare date
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Aplicare PCA
pca = PCA()
pca.fit(data_standardized)

# Extragem proporția varianței explicate
explained_variance_ratio = pca.explained_variance_ratio_

# Afișăm rezultatul
print("Varianta componentelor principale:")
for var in explained_variance_ratio:
    print(var)

#pas 1: citire din csv
df = pd.read_csv('MiscareaNaturala.csv')

#pas2: extragere date numerice
data_numerics = ['FR', 'IM', 'LE', 'LEF', 'LEM','MMR','RS']
data = df[data_numerics]

#pas3: standardizare date
scaler = StandardScaler()
data_standardizate = scaler.fit_transform(data)

#pas4: aplicare PCA()
pca = PCA()
pca.fit(data_standardizate)

#pas5: extragem proportia variatiei
explained_variance_ratio = pca.explained_variance_ratio_

#pas6: afisarea variatiilor:
print('Variantia componentelor principale:')
for var in explained_variance_ratio:
    print(var)

#B.2.
#scorurile asociate instantelor
#scorurile sunt afisare in scoruri.csv

# Scorurile PCA (Proiecția instanțelor în noul spațiu PCA)
scores = pca.transform(data_standardizate)  # Scorurile instanțelor PCA

# Creăm DataFrame-ul pentru scoruri
df_scores = pd.DataFrame(scores, columns=[f'Componenta_{i+1}' for i in range(scores.shape[1])])

# Salvăm scorurile în scoruri.csv
df_scores.to_csv('scoruri.csv', index=False)


#B.3.
#graficul scorurilor in primele doua axe principale

# Extragem primele două componente principale
pc1 = scores[:, 0]  # Prima componentă principală
pc2 = scores[:, 1]  # A doua componentă principală

# Creăm graficul de dispersie (scatter plot)
plt.figure(figsize=(8, 6))
plt.scatter(pc1, pc2, alpha=0.5, edgecolors='k')

# Adăugăm titlu și etichete axelor
plt.title("Graficul scorurilor în primele două axe principale")
plt.xlabel("Componenta Principală 1 (PC1)")
plt.ylabel("Componenta Principală 2 (PC2)")

# Afișăm grila pentru claritate
plt.grid(True, linestyle='--', alpha=0.5)

# Afișăm graficul
plt.show()





















