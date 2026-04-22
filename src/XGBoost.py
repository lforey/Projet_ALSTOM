import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def prepare_xgb_features(df, window_size=50, calib_points=1000):
    trend_vp = df['Vp'].rolling(window_size).median().bfill()
    df['Vp_Residual'] = (df['Vp'] - trend_vp).abs()
    df['Vp_Std'] = df['Vp'].rolling(window_size).std().bfill()
    df['Vp_Speed'] = df['Vp'].diff(window_size).abs().bfill()
    
    trend_ip = df['Ip'].rolling(window_size).median().bfill()
    df['Ip_Residual'] = (df['Ip'] - trend_ip).abs()
    df['Ip_Std'] = df['Ip'].rolling(window_size).std().bfill()
    df['Ip_Speed'] = df['Ip'].diff(window_size).abs().bfill()
    
    df['Ip_Vp_Corr'] = df['Vp'].rolling(window_size).corr(df['Ip']).bfill().fillna(0)
    
    df_feat = df.dropna().copy()
    features_to_scale = [
        'Vp_Residual', 'Vp_Std', 'Vp_Speed', 
        'Ip_Residual', 'Ip_Std', 'Ip_Speed', 
        'Ip_Vp_Corr'
    ]
    
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
    df_feat['Ip_Vp_Corr_scaled'] = scaled_features[:, 6] 
    
    return df_feat

def create_xgb_training_data(all_data_dict, test_id, true_anomaly_times, lookback_sec, pre_arc_target, post_arc_target):
    train_dfs = []
    
    for id_name, t_df in all_data_dict.items():
        if id_name != test_id and id_name in true_anomaly_times:
            arc_time = true_anomaly_times[id_name]
            
            start_time = max(0, arc_time - lookback_sec)
            end_time = arc_time + post_arc_target
            
            train_subset = t_df[(t_df['time'] >= start_time) & (t_df['time'] <= end_time)].copy()
            
            train_subset['Label'] = 0
            train_subset.loc[(train_subset['time'] >= (arc_time - pre_arc_target)), 'Label'] = 1
            
            train_dfs.append(train_subset)
            
    return pd.concat(train_dfs, axis=0)

def train_and_predict_xgb(df_train, test_df, feature_cols, smoothing_window):
    X_train = df_train[feature_cols].values
    y_train = df_train['Label'].values
    X_test = test_df[feature_cols].values
    
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    test_results = test_df.copy()
    test_results['XGB_Proba'] = xgb_model.predict_proba(X_test)[:, 1]
    test_results['Smoothed_Proba'] = test_results['XGB_Proba'].rolling(smoothing_window, min_periods=1).mean()
    
    return test_results

def evaluate_and_plot_xgb(test_results, threshold, true_anomaly_time, test_id):
    alarm_indices = test_results.index[test_results['Smoothed_Proba'] > threshold].tolist()
    
    
    print(f"\n> True anomaly at: {true_anomaly_time:.4f}s")
    
    if alarm_indices:
        ai_alarm_time = test_results['time'].iloc[alarm_indices[0]]
        lead_time = true_anomaly_time - ai_alarm_time
        
        print(f"> First AI alarm raised at: {ai_alarm_time:.4f}s")
        
        
        if ai_alarm_time <= true_anomaly_time - 0.01:
            print(f"Anticipated by {lead_time:.4f} sec.")
        else:
            print(f"No detection in advance")
            
    else:
        print("FAILURE: Probability never exceeded threshold.")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(test_results['time'], test_results['Vp'], label=f'Vp ({test_id})', color='#1f77b4', linewidth=0.8)
    ax1.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2, label="True anomaly start")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc='upper left')
    ax1.set_title(f"XGBoost Supervised Prediction: {test_id}")

    ax2.plot(test_results['time'], test_results['Ip'], label=f'Ip ({test_id})', color='green', linewidth=0.8)
    ax2.axvline(true_anomaly_time, color='orange', linestyle='-', linewidth=2)
    ax2.set_ylabel("Current (A)")
    ax2.legend(loc='upper left')

    ax3.plot(test_results['time'], test_results['Smoothed_Proba'], color='purple', linewidth=1.5, label="Arc Probability")
    ax3.axhline(threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
    ax3.fill_between(test_results['time'], 0, test_results['Smoothed_Proba'], 
                     where=(test_results['Smoothed_Proba'] > threshold), color='purple', alpha=0.3)

    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Probability")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()