
import pandas as pd
import matplotlib.pyplot as plt

Vp = pd.read_csv("MM_B_1_Vp.txt", header=None, names=["Vp"])
x  = pd.read_csv("MM_B_1_x.txt",  header=None, names=["time"])
Vf = pd.read_csv("MM_B_1_Vf.txt", header=None, names=["Vf"])
Ip = pd.read_csv("MM_B_1_Ip.txt", header=None, names=["Ip"])
IR = pd.read_csv("MM_B_1_IR.txt", header=None, names=["IR"])

df = pd.concat([x, Vp, Vf, Ip, IR], axis=1)
df.head()
print(df)



plt.figure(figsize=(14,8))

plt.plot(df["time"], df["Vp"], label="Vp")
plt.plot(df["time"], df["Vf"], label="Vf")
plt.plot(df["time"], df["Ip"], label="Ip")
plt.plot(df["time"], df["IR"], label="IR")

plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signaux électriques – Détection visuelle des arcs")
plt.legend()
plt.grid(True)
plt.show()