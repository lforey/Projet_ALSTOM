import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

def prepare_lstm_features(df, window_size=50, calib_points=1000):
    trend_vp = df['Vp'].rolling(window_size).median().bfill()
    df['Residual'] = (df['Vp'] - trend_vp).abs()
    df['Vp_Std'] = df['Vp'].rolling(window_size).std().bfill()
    df['Vp_Speed'] = df['Vp'].diff(window_size).abs().bfill()
    df['Ip_Vp_Corr'] = df['Vp'].rolling(window_size).corr(df['Ip']).bfill().fillna(0)
    
    df_feat = df.dropna().copy()
    features_to_scale = ['Residual', 'Vp_Std', 'Vp_Speed', 'Ip_Vp_Corr']
    
    calib_data = df_feat[features_to_scale].iloc[:calib_points]
    scaler = StandardScaler()
    scaler.fit(calib_data)
    
    scaled_features = scaler.transform(df_feat[features_to_scale])
    df_feat['Residual_scaled'] = scaled_features[:, 0]
    df_feat['Vp_Std_scaled'] = scaled_features[:, 1]
    df_feat['Vp_Speed_scaled'] = scaled_features[:, 2]
    df_feat['Ip_Vp_Corr_scaled'] = scaled_features[:, 3]
    
    return df_feat

def create_3d_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_labeled_data(t_df, arc_time, lookback_sec, pre_arc_target, post_arc_target):
    start_time = max(0, arc_time - lookback_sec)
    end_time = arc_time + post_arc_target
    
    subset = t_df[(t_df['time'] >= start_time) & (t_df['time'] <= end_time)].copy()
    subset['Label'] = 0
    subset.loc[(subset['time'] >= (arc_time - pre_arc_target)), 'Label'] = 1
    
    return subset

def train_and_predict_lstm(all_data_dict, test_id, true_anomaly_times, feature_cols, params):
    X_train_seqs, y_train_seqs = [], []
    
    # Build 3D sequences per training file to avoid crossing file boundaries
    for id_name, t_df in all_data_dict.items():
        if id_name != test_id and id_name in true_anomaly_times:
            subset = get_labeled_data(
                t_df, true_anomaly_times[id_name], 
                params['lookback'], params['pre_target'], params['post_target']
            )
            X_seq, y_seq = create_3d_sequences(subset[feature_cols], subset['Label'], params['time_steps'])
            X_train_seqs.append(X_seq)
            y_train_seqs.append(y_seq)
            
    X_train = np.vstack(X_train_seqs)
    y_train = np.concatenate(y_train_seqs)
    
    # Handle Imbalance via class weights
    neg, pos = np.bincount(y_train)
    class_weight = {0: 1.0, 1: (neg / pos)}
    
    # --- LSTM ARCHITECTURE ---
    input_layer = Input(shape=(params['time_steps'], len(feature_cols)))
    x = LSTM(32, return_sequences=False)(input_layer)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, y_train, epochs=10, batch_size=64, class_weight=class_weight, verbose=0)
    
    # Inference (Test sequence creation needs dummy labels)
    test_df = all_data_dict[test_id]
    test_df['Dummy'] = 0 
    X_test, _ = create_3d_sequences(test_df[feature_cols], test_df['Dummy'], params['time_steps'])
    
    # Predict and pad the missing initial time steps with zeros
    raw_preds = model.predict(X_test, verbose=0).flatten()
    padded_preds = np.pad(raw_preds, (params['time_steps'], 0), constant_values=0)
    
    test_results = test_df.copy()
    test_results['LSTM_Proba'] = padded_preds
    test_results['Smoothed_Proba'] = test_results['LSTM_Proba'].rolling(params['smoothing'], min_periods=1).mean()
    
    return test_results

def evaluate_and_plot_lstm(test_results, threshold, true_time, pre_arc_target, test_id):
    test_results['Over_Threshold'] = (test_results['Smoothed_Proba'] > threshold).astype(int)
    PERSISTENCE_WINDOW = 50 
    test_results['Alarm_Confidence'] = test_results['Over_Threshold'].rolling(PERSISTENCE_WINDOW).sum()
    alarm_indices = test_results.index[test_results['Alarm_Confidence'] == PERSISTENCE_WINDOW].tolist()
    safe_zone_end_time = true_time - pre_arc_target
    
    print(f"\n  > True anomaly at: {true_time:.4f}s")
    
    if alarm_indices:
        ai_alarm_time = test_results['time'].iloc[alarm_indices[0]]
        lead_time = true_time - ai_alarm_time
        false_alarms = len(test_results[(test_results['time'] < safe_zone_end_time) & 
                                        (test_results['Smoothed_Proba'] > threshold)])
        
        print(f"  > First AI alarm raised at: {ai_alarm_time:.4f}s")
        if ai_alarm_time < safe_zone_end_time:
            print(f"  PREMATURE: Alarm triggered {lead_time:.4f} sec too early.")
        elif ai_alarm_time <= true_time:
            print(f"  SUCCESS: Correctly anticipated by {lead_time:.4f} sec.")
        else:
            print(f"  DELAY: Detected {abs(lead_time):.4f} sec after the arc.")
        print(f"  Total false alarms: {false_alarms}")
    else:
        print("  FAILURE: Probability never exceeded threshold.")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(test_results['time'], test_results['Vp'], label=f'Vp ({test_id})', color='#1f77b4', linewidth=0.8)
    ax1.axvline(true_time, color='orange', linestyle='-', linewidth=2, label="True anomaly start")
    ax1.axvspan(true_time - pre_arc_target, true_time, color='green', alpha=0.3, label="Target Pre-Arc Window")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc='upper left')
    ax1.set_title(f"LSTM Supervised Prediction: {test_id}")

    ax2.plot(test_results['time'], test_results['Smoothed_Proba'], color='darkorange', linewidth=1.5, label="Arc Probability")
    ax2.axhline(threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({threshold*100:.0f}%)")
    ax2.fill_between(test_results['time'], 0, test_results['Smoothed_Proba'], 
                     where=(test_results['Smoothed_Proba'] > threshold), color='darkorange', alpha=0.3)

    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.show()