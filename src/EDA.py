import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_braking_data(braking_id, variables, prefix="MM", base_path="data/"):
    # Construct base time file path
    time_file = f"{base_path}{prefix}_B_{braking_id}_x.txt"
    
    if not os.path.exists(time_file):
        print(f"Error: Missing time file {time_file}")
        return None
        
    df = pd.read_csv(time_file, header=None, names=["time"])
    
    # Iterate through requested variables and concatenate
    for var in variables:
        var_file = f"{base_path}{prefix}_B_{braking_id}_{var}.txt"
        
        if os.path.exists(var_file):
            temp_df = pd.read_csv(var_file, header=None, names=[var])
            df = pd.concat([df, temp_df], axis=1)
        else:
            print(f"Warning: Missing variable file {var_file}")
            
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

    import glob

def get_available_ids(prefix="MM", base_path="data/"):
    """
    Scans the directory for available IDs matching the prefix format (e.g., MM_B_1_Vp.txt)
    """
    search_pattern = f"{base_path}{prefix}_B_*_Vp.txt"
    file_list = sorted(glob.glob(search_pattern))
    
    available_ids = []
    for file_path in file_list:
        filename = os.path.basename(file_path)
        # For 'MM_B_1_Vp.txt', split('_') gives ['MM', 'B', '1', 'Vp.txt']
        # The ID is at index 2
        b_id = filename.split('_')[2]
        available_ids.append(b_id)
        
    return available_ids