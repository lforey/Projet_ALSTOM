import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def prepare_iforest_features(df, window_size=50, calib_points=1000):
    """Calculates features and applies local calibration for Vp and Ip."""
    # Feature Engineering
    trend_vp = df['Vp'].rolling(window_size).median()
    df['Vp_Residual'] = (df['Vp'] - trend_vp).abs() 
    df['Vp_Std'] = df['Vp'].rolling(window_size).std()
    df['Vp_Speed'] = df['Vp'].diff().abs().fillna(0) 
    trend_ip = df['Ip'].rolling(window_size).median()
    df['Ip_Residual'] = (df['Ip'] - trend_ip).abs()
    df['Ip_Std'] = df['Ip'].rolling(window_size).std()
    df['Ip_Speed'] = df['Ip'].diff().abs().fillna(0)
    
    df_feat = df.dropna().copy()
    
    # Local Normalization on the first 'calib_points'
    features_to_scale = ['Vp_Residual', 'Vp_Std', 'Vp_Speed', 'Ip_Residual', 'Ip_Std', 'Ip_Speed']
    calib_data = df_feat[features_to_scale].iloc[:calib_points]
    
    scaler = StandardScaler()
    scaler.fit(calib_data)
    
    # Apply to the whole file
    scaled_features = scaler.transform(df_feat[features_to_scale])
    df_feat['Vp_Residual_scaled'] = scaled_features[:, 0]
    df_feat['Vp_Std_scaled'] = scaled_features[:, 1]
    df_feat['Vp_Speed_scaled'] = scaled_features[:, 2]
    df_feat['Ip_Residual_scaled'] = scaled_features[:, 3]
    df_feat['Ip_Std_scaled'] = scaled_features[:, 4]
    df_feat['Ip_Speed_scaled'] = scaled_features[:, 5]
    
    return df_feat

def train_and_predict_loocv(train_dfs, test_df, feature_cols, num_train_points, contamination=0.005):
    """Trains Isolation Forest on train_dfs and predicts on test_df using specified features."""
    train_features_list = [df[feature_cols].iloc[:num_train_points] for df in train_dfs]
    X_train = pd.concat(train_features_list, axis=0).values
    X_test = test_df[feature_cols].values
    
    # Train model
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_train)
    
    # Calculate scores
    train_scores = -model.decision_function(X_train)
    test_scores = -model.decision_function(X_test)
    
    # Dynamic Threshold calculation
    smoothed_train = pd.Series(train_scores).rolling(5).mean().fillna(pd.Series(train_scores))
    max_train_score = smoothed_train.max()
    
    # Add results to test dataframe
    test_results = test_df.copy()
    test_results['AI_Score'] = test_scores
    test_results['Smoothed_Score'] = test_results['AI_Score'].rolling(5).mean().fillna(test_results['AI_Score'])
    
    return test_results, max_train_score

def evaluate_and_plot(test_results, threshold, true_anomaly_time, test_id):
    """Calculates metrics and plots the results """
    alarm_indices = test_results.index[test_results['Smoothed_Score'] > threshold].tolist()
 
    
    print(f"\n  > True anomaly at: {true_anomaly_time:.4f}s")
    
    if alarm_indices:
        ai_alarm_time = test_results['time'].iloc[alarm_indices[0]]
        lead_time = true_anomaly_time - ai_alarm_time
        
        print(f"  > First AI alarm raised at: {ai_alarm_time:.4f}s")
        if ai_alarm_time <= true_anomaly_time - 0.01:
            print(f"Anticipated by {lead_time:.4f} sec.")
        else:
            print(f"No detection in advance")
            

    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Vp Plot
    ax1.plot(test_results['time'], test_results['Vp'], label=f'Vp ({test_id})', color='blue', alpha=0.6)
    ax1.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2, label="True anomaly start")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend()
    
    # Ip Plot
    ax2.plot(test_results['time'], test_results['Ip'], label=f'Ip ({test_id})', color='green', alpha=0.6)
    ax2.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2)
    ax2.set_ylabel("Current (A)")
    ax2.legend()
    
    # Anomaly Score Plot
    ax3.plot(test_results['time'], test_results['Smoothed_Score'], color='purple', linewidth=1, label="Anomaly Score")
    ax3.axhline(threshold, color='red', linestyle='--', label=f"Threshold ({threshold:.2f})")
    ax3.set_ylabel("Anomaly Deviation")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper left')
    
    plt.suptitle(f"Isolation Forest Backtest - Test File: {test_id}")
    plt.tight_layout()
    plt.show()