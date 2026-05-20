#!/usr/bin/env python3
"""
train_model.py — Módulo 3
Entrena y compara 3 modelos de clasificación.
Guarda el mejor modelo (Random Forest) con joblib.
Entregable E-04
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # sin pantalla gráfica en Ubuntu Server

# ── Configuración ────────────────────────────────────────────────
DATASET_PATH  = 'data/dataset.csv'
MODEL_DIR     = 'models/'
RANDOM_STATE  = 42
TEST_SIZE     = 0.30
FEATURE_COLS  = [
    'total_pkts', 'tcp_pkts', 'udp_pkts', 'other_pkts',
    'unique_dports_count', 'syn_ratio', 'avg_pkt_size',
    'duration_sec', 'bytes_per_sec', 'port_scan_score',
    'small_syn_score', 'potential_flood', 'potential_scan'
]

def load_dataset():
    print("[*] Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Filas: {len(df)} · Columnas: {len(df.columns)}")
    print(f"[*] Clases:\n{df['label'].value_counts()}\n")

    X = df[FEATURE_COLS].values
    y = (df['label'] == 'attack').astype(int).values  # 1=attack, 0=normal
    return X, y

def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,           # mantiene proporción de clases — RF-08
        random_state=RANDOM_STATE
    )

def evaluate_model(name, model, X_test, y_test):
    """Evalúa un modelo y retorna sus métricas"""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report  = classification_report(y_test, y_pred,
                                     target_names=['normal', 'attack'],
                                     output_dict=True)
    auc     = roc_auc_score(y_test, y_proba)

    recall_attack    = report['attack']['recall']
    precision_attack = report['attack']['precision']
    f1_attack        = report['attack']['f1-score']
    accuracy         = report['accuracy']

    print(f"\n{'='*55}")
    print(f"  Modelo: {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                 target_names=['normal', 'attack']))
    print(f"  AUC-ROC: {auc:.4f}")

    # Verificar criterios bloqueantes
    ok_recall = "✓" if recall_attack >= 0.90 else "✗ FALLO"
    ok_auc    = "✓" if auc >= 0.95 else "✗ FALLO"
    print(f"\n  Recall attack:  {recall_attack:.4f}  {ok_recall} (min 0.90)")
    print(f"  AUC-ROC:        {auc:.4f}  {ok_auc}    (min 0.95)")

    return {
        'modelo':    name,
        'accuracy':  round(accuracy, 4),
        'recall':    round(recall_attack, 4),
        'precision': round(precision_attack, 4),
        'f1':        round(f1_attack, 4),
        'auc_roc':   round(auc, 4),
        'cumple':    recall_attack >= 0.90 and auc >= 0.95
    }

def plot_confusion_matrix(model, X_test, y_test, name):
    """Guarda imagen de la matriz de confusión"""
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['normal', 'attack'])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Matriz de Confusión — {name}')
    path = f'informe/confusion_matrix_{name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [✓] Matriz guardada: {path}")

def plot_feature_importance(model, name):
    """Guarda gráfico de importancia de features (solo para RF y GBT)"""
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(FEATURE_COLS)),
           importances[indices], color='steelblue')
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in indices],
                        rotation=45, ha='right', fontsize=9)
    ax.set_title(f'Importancia de Features — {name}')
    ax.set_ylabel('Importancia')
    path = f'informe/feature_importance_{name.lower().replace(" ","_")}.png'
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [✓] Feature importance guardada: {path}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('informe', exist_ok=True)

    # ── Cargar y dividir datos ───────────────────────────────────
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[*] Train: {len(X_train)} muestras · Test: {len(X_test)} muestras")

    # ── Escalar features ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, f'{MODEL_DIR}scaler.joblib')
    print("[✓] scaler.joblib guardado")

    # ── Definir los 3 modelos — RF-09 ───────────────────────────
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        ),
    }

    # ── Entrenar y evaluar cada modelo ──────────────────────────
    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        print(f"\n[*] Entrenando {nombre}...")
        modelo.fit(X_train_sc, y_train)
        metricas = evaluate_model(nombre, modelo, X_test_sc, y_test)
        resultados.append(metricas)
        modelos_entrenados[nombre] = modelo
        plot_confusion_matrix(modelo, X_test_sc, y_test, nombre)
        plot_feature_importance(modelo, nombre)

    # ── Tabla comparativa — CA-07 ────────────────────────────────
    print(f"\n{'='*65}")
    print("  TABLA COMPARATIVA DE MODELOS (CA-07)")
    print(f"{'='*65}")
    df_res = pd.DataFrame(resultados)
    print(df_res.to_string(index=False))
    df_res.to_csv('informe/comparativa_modelos.csv', index=False)
    print("\n[✓] comparativa_modelos.csv guardado en informe/")

    # ── Guardar el mejor modelo (Random Forest) — RF-08 ─────────
    mejor = modelos_entrenados['Random Forest']
    model_path = f'{MODEL_DIR}firewall_ai_model.joblib'
    joblib.dump(mejor, model_path)
    print(f"\n[✓] Modelo principal guardado: {model_path}")

    # ── Verificación de criterios bloqueantes ────────────────────
    rf_result = next(r for r in resultados if r['modelo'] == 'Random Forest')
    print(f"\n{'='*55}")
    print("  VERIFICACIÓN CRITERIOS BLOQUEANTES")
    print(f"{'='*55}")
    print(f"  CA-05 Recall attack >= 0.90 : {rf_result['recall']}  {'✓ OK' if rf_result['recall'] >= 0.90 else '✗ FALLO'}")
    print(f"  CA-06 AUC-ROC   >= 0.95     : {rf_result['auc_roc']} {'✓ OK' if rf_result['auc_roc'] >= 0.95 else '✗ FALLO'}")

    if rf_result['cumple']:
        print("\n  ✓ Modelo aprobado — listo para el Módulo 4")
    else:
        print("\n  ✗ ATENCIÓN: el modelo no cumple los criterios mínimos")
        print("    → Captura más tráfico y regenera el dataset")

if __name__ == '__main__':
    main()
