"""
model.py
Bidirectional LSTM + Bahdanau Attention + Bayesian Hyperparameter Optimization
"""

import numpy as np
import pandas as pd
import os
import json
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout,
    Layer, Multiply, Softmax, Lambda, Flatten, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']


# ─────────────────────────────────────────────────────────────
# BAHDANAU ATTENTION LAYER
# ─────────────────────────────────────────────────────────────
class BahdanauAttention(Layer):
    """Soft attention over LSTM sequence output."""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, use_bias=False)
        self.V = Dense(1, use_bias=False)

    def call(self, encoder_output):
        # encoder_output: (batch, timesteps, features)
        score = self.V(tf.nn.tanh(self.W(encoder_output)))   # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)               # (batch, T, 1)
        context = weights * encoder_output                    # (batch, T, features)
        context = tf.reduce_sum(context, axis=1)             # (batch, features)
        return context, tf.squeeze(weights, axis=-1)

    def get_config(self):
        config = super().get_config()
        return config


# ─────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────
def build_model(seq_len, n_features, lstm1=128, lstm2=64,
                dropout=0.2, lr=0.001, attn_units=64):
    inputs = Input(shape=(seq_len, n_features), name='input')

    # BiLSTM Layer 1
    x = Bidirectional(LSTM(lstm1, return_sequences=True, name='lstm1'),
                      name='bilstm1')(inputs)
    x = Dropout(dropout, name='drop1')(x)

    # Bahdanau Attention
    attn_layer = BahdanauAttention(units=attn_units, name='attention')
    context, attn_weights = attn_layer(x)

    # BiLSTM Layer 2 - uses context
    x2 = Bidirectional(LSTM(lstm2, return_sequences=False, name='lstm2'),
                       name='bilstm2')(x)
    x2 = Dropout(dropout, name='drop2')(x2)

    # Merge context + lstm2 output
    merged = tf.keras.layers.Concatenate(name='merge')([context, x2])

    # Dense head
    x3 = Dense(64, activation='relu', name='dense1')(merged)
    x3 = BatchNormalization(name='bn')(x3)
    x3 = Dropout(dropout / 2, name='drop3')(x3)
    x3 = Dense(32, activation='relu', name='dense2')(x3)

    # Output: predict all 8 blood groups
    output = Dense(len(BLOOD_GROUPS), activation='linear', name='output')(x3)

    model = Model(inputs=inputs, outputs=output, name='HemaForecast_BiLSTM_Attention')
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model


# ─────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────
def prepare_sequences(df, seq_len=12):
    """Create sliding window sequences for LSTM training."""
    features = BLOOD_GROUPS + ['month', 'total_demand']
    data = df[features].values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i + seq_len])
        y.append(data_scaled[i + seq_len, :len(BLOOD_GROUPS)])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler, features


def train_test_split_time(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def train_model(df, seq_len=12, epochs=100, batch_size=32,
                lstm1=128, lstm2=64, dropout=0.22, lr=0.0018):
    print("[*] Preparing sequences...")
    X, y, scaler, features = prepare_sequences(df, seq_len)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)

    print(f"[*] Train: {X_train.shape}, Test: {X_test.shape}")

    n_features = X_train.shape[2]
    model = build_model(seq_len, n_features, lstm1, lstm2, dropout, lr)
    model.summary()

    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
    ]

    print("[*] Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, scaler, n_features)
    print(f"[OK] MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}")

    # Save scaler and metadata
    joblib.dump(scaler, 'models/scaler.pkl')
    meta = {
        'seq_len': seq_len, 'n_features': n_features, 'features': features,
        'blood_groups': BLOOD_GROUPS, 'epochs_run': len(history.history['loss']),
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'metrics': metrics,
        'hyperparams': {'lstm1': lstm1, 'lstm2': lstm2, 'dropout': dropout, 'lr': lr, 'batch': batch_size}
    }
    with open('models/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("[OK] Model saved to models/best_model.keras")
    return model, scaler, history, metrics


def evaluate_model(y_true, y_pred, scaler, n_features):
    # Inverse transform (only blood group columns)
    dummy = np.zeros((y_true.shape[0], n_features))
    dummy[:, :len(BLOOD_GROUPS)] = y_true
    y_true_inv = scaler.inverse_transform(dummy)[:, :len(BLOOD_GROUPS)]

    dummy2 = np.zeros((y_pred.shape[0], n_features))
    dummy2[:, :len(BLOOD_GROUPS)] = y_pred
    y_pred_inv = scaler.inverse_transform(dummy2)[:, :len(BLOOD_GROUPS)]

    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-8))) * 100
    r2   = r2_score(y_true_inv.flatten(), y_pred_inv.flatten())

    return {'mae': round(mae, 4), 'rmse': round(rmse, 4),
            'mape': round(mape, 4), 'r2': round(r2, 4)}


# ─────────────────────────────────────────────────────────────
# FORECASTING
# ─────────────────────────────────────────────────────────────
def forecast_future(model, df, scaler, seq_len=12, n_months=6):
    """Predict next n_months blood demand for all groups."""
    features = BLOOD_GROUPS + ['month', 'total_demand']
    data = df[features].values.astype(np.float32)
    data_scaled = scaler.transform(data)

    seq = data_scaled[-seq_len:].copy()
    predictions = []
    last_date = pd.to_datetime(df['date'].iloc[-1])

    for i in range(n_months):
        x_in = seq[-seq_len:].reshape(1, seq_len, -1)
        pred = model.predict(x_in, verbose=0)[0]

        # Build next step with predicted blood groups
        next_month_num = (last_date.month + i) % 12 + 1
        next_total = pred.sum()
        next_row = np.zeros(len(features))
        next_row[:len(BLOOD_GROUPS)] = pred
        next_row[len(BLOOD_GROUPS)] = next_month_num / 12.0
        next_row[len(BLOOD_GROUPS) + 1] = next_total

        seq = np.vstack([seq, next_row])

        # Inverse transform
        dummy = np.zeros((1, len(features)))
        dummy[0, :len(BLOOD_GROUPS)] = pred
        inv = scaler.inverse_transform(dummy)[0, :len(BLOOD_GROUPS)]
        inv = np.maximum(inv, 0).astype(int)

        forecast_date = last_date + pd.DateOffset(months=i + 1)
        row = {'date': forecast_date.strftime('%Y-%m')}
        row.update({bg: int(inv[j]) for j, bg in enumerate(BLOOD_GROUPS)})
        row['total'] = int(inv.sum())
        predictions.append(row)

    return predictions


def get_attention_weights(model, df, scaler, seq_len=12):
    """Get attention weights for the last sequence."""
    features = BLOOD_GROUPS + ['month', 'total_demand']
    data = df[features].values.astype(np.float32)
    data_scaled = scaler.transform(data)
    seq = data_scaled[-seq_len:].reshape(1, seq_len, -1)

    # Build attention sub-model
    attn_model = Model(
        inputs=model.input,
        outputs=model.get_layer('attention').output
    )
    _, weights = attn_model.predict(seq, verbose=0)
    return weights[0].tolist()


# ─────────────────────────────────────────────────────────────
# BAYESIAN OPTIMIZATION (Optuna)
# ─────────────────────────────────────────────────────────────
def run_bayesian_optimization(df, n_trials=15):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[WARN] Optuna not installed — skipping optimization")
        return {'lstm1': 128, 'lstm2': 64, 'dropout': 0.22, 'lr': 0.0018, 'batch': 32}

    X, y, scaler, features = prepare_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)
    n_features = X_train.shape[2]

    def objective(trial):
        lstm1   = trial.suggest_categorical('lstm1', [64, 128, 256])
        lstm2   = trial.suggest_categorical('lstm2', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr      = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch   = trial.suggest_categorical('batch', [16, 32, 64])

        m = build_model(12, n_features, lstm1, lstm2, dropout, lr)
        cb = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
        m.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=40, batch_size=batch, callbacks=cb, verbose=0)
        loss = m.evaluate(X_test, y_test, verbose=0)[0]
        return loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"[OK] Best params: {best}")
    return best


if __name__ == '__main__':
    df = pd.read_csv('data/blood_demand.csv', parse_dates=['date'])
    df['month'] = df['date'].dt.month
    model, scaler, history, metrics = train_model(df)
    print("[DONE] Training complete:", metrics)
