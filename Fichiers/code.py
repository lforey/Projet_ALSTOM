
import pandas as pd
import matplotlib.pyplot as plt

Vp1 = pd.read_csv("MM_B_1_Vp.txt", header=None, names=["Vp"])
x1  = pd.read_csv("MM_B_1_x.txt",  header=None, names=["time"])
Vf1 = pd.read_csv("MM_B_1_Vf.txt", header=None, names=["Vf"])
Ip1 = pd.read_csv("MM_B_1_Ip.txt", header=None, names=["Ip"])
IR1 = pd.read_csv("MM_B_1_IR.txt", header=None, names=["IR"])

df1 = pd.concat([x1, Vp1, Vf1, Ip1, IR1], axis=1)
df1.head()
print(df1)


plt.figure(figsize=(14,8))

plt.plot(df1["time"], df1["Vp"], label="Vp")
plt.plot(df1["time"], df1["Vf"], label="Vf")
plt.plot(df1["time"], df1["Ip"], label="Ip")
plt.plot(df1["time"], df1["IR"], label="IR")

plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signaux électriques – Détection visuelle des arcs")
plt.legend()
plt.grid(True)
plt.show()

window_size = 50 

# Calcul de la volatilité glissante sur Vp et Ip
df1['Vp_std'] = df1['Vp'].rolling(window=window_size).std()
df1['Ip_std'] = df1['Ip'].rolling(window=window_size).std()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df1["time"], df1["Vp_std"], label="Instabilité Tension (Rolling Std)", color='purple')
ax.set_title("Précurseur potentiel : Instabilité du signal")
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Écart-type glissant")
ax.legend()
ax.grid(True)

# Seuil d'alerte hypothétique
ax.axhline(y=df1['Vp_std'].mean() + 3*df1['Vp_std'].std(), color='r', linestyle='--', label='Seuil Alerte')
plt.show()





