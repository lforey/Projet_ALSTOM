import pandas as pd
import matplotlib.pyplot as plt

def load_braking_data(braking_id, variables, base_path="Fichiers/"):
    # Load time array first
    df = pd.read_csv(f"{base_path}MM_B_{braking_id}_x.txt", header=None, names=["time"])
    
    # Append requested variables
    for var in variables:
        temp_df = pd.read_csv(f"{base_path}MM_B_{braking_id}_{var}.txt", header=None, names=[var])
        df = pd.concat([df, temp_df], axis=1)
        
    return df

def plot_all_signals(df, title="Electrical Signals - Visual Arc Detection"):
    plt.figure(figsize=(14, 8))
    
    # Plot all columns against time
    for col in df.columns:
        if col != "time":
            plt.plot(df["time"], df[col], label=col)
            
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_voltage_current(df, braking_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Voltage
    ax1.plot(df['time'], df['Vp'], color='tab:blue', label='Voltage Vp')
    ax1.set_title(f"Record #{braking_id}: Visual Arc Detection")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # Plot Current
    ax2.plot(df['time'], df['Ip'], color='tab:red', label='Current Ip')
    ax2.set_ylabel("Current (A)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()