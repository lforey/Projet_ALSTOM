import matplotlib.pyplot as plt
import numpy as np 

def detect_instabilities(df, window_size=50, sigma_threshold=5):
    """Calculates rolling standard deviation of Vp to detect arc instabilities."""
    vp_std = df['Vp'].rolling(window=window_size).std()
    
    median_std = vp_std.median()
    sigma_std = vp_std.std()
    threshold = median_std + (sigma_threshold * sigma_std)
    
    unstable_indices = df.index[vp_std > threshold].tolist()
    return unstable_indices

def plot_instabilities(df, unstable_indices, braking_id):
    """Plots the Vp signal and highlights unstable points in red."""
    plt.figure(figsize=(12, 4))
    plt.plot(df['time'], df['Vp'], label='Vp Signal', color='blue', alpha=0.7)
    
    if unstable_indices:
        plt.scatter(df['time'].iloc[unstable_indices], df['Vp'].iloc[unstable_indices], 
                    color='red', label='Detected Instabilities', s=10)
        
    plt.title(f"Anomaly Detection for Record #{braking_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Vp (V)")
    plt.legend()
    plt.show()

def extract_correlations(df, unstable_indices, pre_arc_window=10000):
    """Extracts Vp/Ip correlations for normal and pre-arc phases."""
    normal_corrs = []
    pre_arc_corrs = []
    
    # CASE 1: ARC DETECTED
    if unstable_indices:
        onset_idx = unstable_indices[0]
        
        # Danger Zone (Pre-Arc)
        if onset_idx > pre_arc_window:
            danger_segment = df.iloc[onset_idx - pre_arc_window - 3000 : onset_idx - 3000]
            if danger_segment['Vp'].std() > 0.1 and danger_segment['Ip'].std() > 0.1:
                pre_arc_corrs.append(danger_segment['Vp'].corr(danger_segment['Ip']))
        
        # Safe Zone (Normal)
        safe_zone_end = onset_idx - 35000
        for i in range(50, safe_zone_end, 200):
            normal_segment = df.iloc[i : i + 200]
            if normal_segment['Vp'].std() > 0.1 and normal_segment['Ip'].std() > 0.1:
                normal_corrs.append(normal_segment['Vp'].corr(normal_segment['Ip']))
                
    # CASE 2: NO ARC DETECTED
    else:
        mid = len(df) // 2
        normal_segment = df.iloc[mid : mid + pre_arc_window]
        if normal_segment['Vp'].std() > 0.1 and normal_segment['Ip'].std() > 0.1:
            normal_corrs.append(normal_segment['Vp'].corr(normal_segment['Ip']))

    # Clean NaNs
    normal_corrs = [c for c in normal_corrs if not np.isnan(c)]
    pre_arc_corrs = [c for c in pre_arc_corrs if not np.isnan(c)]
    
    return normal_corrs, pre_arc_corrs

def plot_correlation_boxplot(normal_corrs, pre_arc_corrs):
    """Plots a boxplot comparing normal and pre-arc correlations."""
    plt.figure(figsize=(10, 6))
    data, labels = [], []
    
    if normal_corrs:
        data.append(normal_corrs)
        labels.append('Normal')
    if pre_arc_corrs:
        data.append(pre_arc_corrs)
        labels.append('Pre-Arc')

    if data:
        plt.boxplot(data, labels=labels)
        plt.grid(True)
        plt.ylabel("Vp/Ip Pearson Correlation")
        plt.title("Vp-Ip Signature Comparison")
        plt.show()