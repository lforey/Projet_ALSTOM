import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def prepare_ae_features(df, window_size=50, calib_points=1000):
    trend_vp = df['Vp'].rolling(window_size).median().bfill()
    df['Residual'] = (df['Vp'] - trend_vp).abs()
    df['Vp_Std'] = df['Vp'].rolling(window_size).std().bfill()
    df['Vp_Speed'] = df['Vp'].diff(window_size).abs().bfill()
    
    trend_ip = df['Ip'].rolling(window_size).median().bfill()
    df['Ip_Residual'] = (df['Ip'] - trend_ip).abs()
    df['Ip_Std'] = df['Ip'].rolling(window_size).std().bfill()
    df['Ip_Speed'] = df['Ip'].diff(window_size).abs().bfill()
    
    df_feat = df.dropna().copy()
    features_to_scale = ['Residual', 'Vp_Std', 'Vp_Speed', 'Ip_Residual', 'Ip_Std', 'Ip_Speed']
    
    calib_data = df_feat[features_to_scale].iloc[:calib_points]
    scaler = StandardScaler()
    scaler.fit(calib_data)
    
    scaled_features = scaler.transform(df_feat[features_to_scale])
    df_feat['Vp_Residual_scaled'] = scaled_features[:, 0]
    df_feat['Vp_Std_scaled'] = scaled_features[:, 1]
    df_feat['Vp_Speed_scaled'] = scaled_features[:, 2]
    df_feat['Ip_Residual_scaled'] = scaled_features[:, 3]
    df_feat['Ip_Std_scaled'] = scaled_features[:, 4]
    df_feat['Ip_Speed_scaled'] = scaled_features[:, 5]
    
    return df_feat

def train_and_predict_ae(train_dfs, test_df, feature_cols, num_train_points, threshold_margin=1.2):
    train_features_list = [df[feature_cols].iloc[:num_train_points] for df in train_dfs]
    X_train = pd.concat(train_features_list, axis=0).values
    X_test = test_df[feature_cols].values
    
    input_dim = X_train.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(8, activation="relu")(input_layer)
    encoder = Dense(4, activation="relu")(encoder)
    bottleneck = Dense(2, activation="relu")(encoder) 
    decoder = Dense(4, activation="relu")(bottleneck)
    decoder = Dense(8, activation="relu")(decoder)
    output_layer = Dense(input_dim, activation="linear")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, verbose=0)
    
    train_reconstructions = autoencoder.predict(X_train, verbose=0)
    train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
    smoothed_train_mse = pd.Series(train_mse).rolling(500, min_periods=1).mean()
    base_error = np.percentile(smoothed_train_mse.dropna(), 99.9)
    threshold = base_error * threshold_margin
    
    test_reconstructions = autoencoder.predict(X_test, verbose=0)
    test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)
    
    test_results = test_df.copy()
    test_results['AI_Score'] = test_mse
    test_results['Smoothed_Score'] = test_results['AI_Score'].rolling(500, min_periods=1).mean()
    
    return test_results, threshold

def evaluate_and_plot_ae(test_results, threshold, true_anomaly_time, test_id):
    alarm_indices = test_results.index[test_results['Smoothed_Score'] > threshold].tolist()
    safe_zone_end_time = true_anomaly_time - 1 
    
    print(f"\n  > True anomaly at: {true_anomaly_time:.4f}s")
    
    if alarm_indices:
        ai_alarm_time = test_results['time'].iloc[alarm_indices[0]]
        lead_time = true_anomaly_time - ai_alarm_time
        false_alarms = len(test_results[(test_results['time'] < safe_zone_end_time) & 
                                        (test_results['Smoothed_Score'] > threshold)])
        print(f"  > First AI alarm raised at: {ai_alarm_time:.4f}s")
        if ai_alarm_time <= true_anomaly_time - 0.01:
            print(f"Anticipated by {lead_time:.4f} sec.")
        else:
            print(f"No detection in advance")
            
    else:
        print("  FAILURE: AI never exceeded the threshold.")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(test_results['time'], test_results['Vp'], label=f'Vp ({test_id})', color='#1f77b4', linewidth=0.8)
    ax1.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2, label="True anomaly start")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc='upper left')
    ax1.set_title(f"Autoencoder Backtest: {test_id}")

    ax2.plot(test_results['time'], test_results['Ip'], label=f'Ip ({test_id})', color='green', linewidth=0.8)
    ax2.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2)
    ax2.set_ylabel("Current (A)")
    ax2.legend(loc='upper left')

    ax3.plot(test_results['time'], test_results['Smoothed_Score'], color='crimson', linewidth=1, label="Reconstruction Error (MSE)")
    ax3.axhline(threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({threshold:.2e})")
    ax3.fill_between(test_results['time'], 0, test_results['Smoothed_Score'], 
                     where=(test_results['Smoothed_Score'] > threshold), color='red', alpha=0.5)

    ax3.set_ylim(0, threshold * 4) 
    ax3.set_ylabel("MSE (Anomaly Score)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def prepare_raw_ae_features(df, calib_points=1000):
    df_feat = df.dropna().copy()
    raw_features = ['Vp', 'Ip', 'IR', 'Vf']
    
    calib_data = df_feat[raw_features].iloc[:calib_points]
    scaler = StandardScaler()
    scaler.fit(calib_data)
    
    scaled_features = scaler.transform(df_feat[raw_features])
    df_feat['Vp_scaled'] = scaled_features[:, 0]
    df_feat['Ip_scaled'] = scaled_features[:, 1]
    df_feat['IR_scaled'] = scaled_features[:, 2]
    df_feat['Vf_scaled'] = scaled_features[:, 3]
    
    return df_feat

def train_and_predict_raw_ae(train_dfs, test_df, feature_cols, num_train_points, threshold_margin=1.2):
    train_features_list = [df[feature_cols].iloc[:num_train_points] for df in train_dfs]
    X_train = pd.concat(train_features_list, axis=0).values
    X_test = test_df[feature_cols].values
    
    input_dim = X_train.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(8, activation="relu")(input_layer)
    encoder = Dense(4, activation="relu")(encoder)
    bottleneck = Dense(2, activation="relu")(encoder) 
    decoder = Dense(4, activation="relu")(bottleneck)
    decoder = Dense(8, activation="relu")(decoder)
    output_layer = Dense(input_dim, activation="linear")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, verbose=0)
    
    train_reconstructions = autoencoder.predict(X_train, verbose=0)
    train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
    smoothed_train_mse = pd.Series(train_mse).rolling(500, min_periods=1).mean()
    base_error = np.percentile(smoothed_train_mse.dropna(), 99.9)
    threshold = base_error * threshold_margin
    
    test_reconstructions = autoencoder.predict(X_test, verbose=0)
    test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)
    
    test_results = test_df.copy()
    test_results['AI_Score'] = test_mse
    test_results['Smoothed_Score'] = test_results['AI_Score'].rolling(500, min_periods=1).mean()
    
    return test_results, threshold

def evaluate_and_plot_raw_ae(test_results, threshold, true_anomaly_time, test_id):
    alarm_indices = test_results.index[test_results['Smoothed_Score'] > threshold].tolist()
    safe_zone_end_time = true_anomaly_time - 1 
    
    print(f"\n  > True anomaly at: {true_anomaly_time:.4f}s")
    
    if alarm_indices:
        ai_alarm_time = test_results['time'].iloc[alarm_indices[0]] 
        lead_time = true_anomaly_time - ai_alarm_time
        false_alarms = len(test_results[(test_results['time'] < safe_zone_end_time) & 
                                        (test_results['Smoothed_Score'] > threshold)])
        
        print(f"  > First AI alarm raised at: {ai_alarm_time:.4f}s")
        if ai_alarm_time <= true_anomaly_time - 0.01:
            print(f"Anticipated by {lead_time:.4f} sec.")
        else:
            print(f"No detection in advance")
    else:
        print("  FAILURE: AI never exceeded the threshold.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(test_results['time'], test_results['Vp'], label=f'Raw Vp ({test_id})', color='#1f77b4', linewidth=0.8)
    ax1.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2, label="True anomaly start")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc='upper left')
    ax1.set_title(f"Autoencoder Backtest: {test_id} (Raw Signals: Vp, Ip, IR, Vf)")

    ax2.plot(test_results['time'], test_results['Smoothed_Score'], color='crimson', linewidth=1, label="MSE")
    ax2.axhline(threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({threshold:.2e})")
    ax2.fill_between(test_results['time'], 0, test_results['Smoothed_Score'], 
                     where=(test_results['Smoothed_Score'] > threshold), color='red', alpha=0.5)

    ax2.set_ylim(0, threshold * 4) 
    ax2.set_ylabel("Anomaly Score")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()