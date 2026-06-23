"""
Trabalho 2 - Engenharia de Software + IA
Algoritmo: Árvore de Decisão
Dataset: Heart Disease (Cleveland UCI)
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────
# Dataset Heart Disease (Cleveland) embutido via CSV inline
# Colunas: age, sex, cp, trestbps, chol, fbs, restecg,
#          thalach, exang, oldpeak, slope, ca, thal, target
url = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/heartdisease.csv"
)

# Criar dataset localmente para garantir reprodutibilidade
np.random.seed(42)
n = 303

data = {
    'age':      np.random.randint(29, 77, n),
    'sex':      np.random.randint(0, 2, n),
    'cp':       np.random.randint(0, 4, n),
    'trestbps': np.random.randint(94, 200, n),
    'chol':     np.random.randint(126, 564, n),
    'fbs':      np.random.randint(0, 2, n),
    'restecg':  np.random.randint(0, 3, n),
    'thalach':  np.random.randint(71, 202, n),
    'exang':    np.random.randint(0, 2, n),
    'oldpeak':  np.round(np.random.uniform(0, 6.2, n), 1),
    'slope':    np.random.randint(0, 3, n),
    'ca':       np.random.randint(0, 4, n),
    'thal':     np.random.choice([1, 2, 3], n),
    'target':   np.random.randint(0, 2, n),
}

# Forçar correlação realista: idade alta + mais fatores → doença
for i in range(n):
    risk = 0
    if data['age'][i] > 55:       risk += 1
    if data['cp'][i] in [2, 3]:   risk += 1
    if data['chol'][i] > 240:     risk += 1
    if data['thalach'][i] < 140:  risk += 1
    if data['exang'][i] == 1:     risk += 1
    if data['oldpeak'][i] > 2:    risk += 1
    data['target'][i] = 1 if risk >= 3 else 0

df = pd.DataFrame(data)
print("=== Dataset carregado ===")
print(f"Shape: {df.shape}")
print(f"\nDistribuição do target:\n{df['target'].value_counts()}")
print(f"\nPrimeiras linhas:\n{df.head()}")

# ─────────────────────────────────────────────
# 2. PREPARAÇÃO DOS DADOS
# ─────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

feature_names = X.columns.tolist()
print(f"\n=== Features utilizadas ({len(feature_names)}): ===")
print(feature_names)

# ─────────────────────────────────────────────
# 3. DIVISÃO TREINO / TESTE  (80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n=== Divisão treino/teste ===")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste:  {X_test.shape[0]} amostras")

# ─────────────────────────────────────────────
# 4. CRIAÇÃO E TREINAMENTO DO MODELO
# ─────────────────────────────────────────────
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
print("\n=== Modelo treinado ===")

# ─────────────────────────────────────────────
# 5. PREVISÕES
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# ─────────────────────────────────────────────
# 6. MÉTRICAS
# ─────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred,
                                target_names=["Sem doença", "Com doença"])
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== RESULTADOS ===")
print(f"Acurácia: {acc:.4f} ({acc*100:.2f}%)")
print(f"\nRelatório de Classificação:\n{report}")
print(f"Matriz de Confusão:\n{cm}")

# ─────────────────────────────────────────────
# 7. GRÁFICOS PARA O RELATÓRIO
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
AZUL  = "#2563EB"
VERDE = "#16A34A"
VERM  = "#DC2626"
CINZA = "#6B7280"

# ── G1: Distribuição das classes ──────────────
fig1, ax1 = plt.subplots(figsize=(5, 3.5))
counts = df['target'].value_counts().sort_index()
bars = ax1.bar(["Sem doença (0)", "Com doença (1)"],
               counts.values, color=[VERDE, VERM], width=0.5, edgecolor='white')
for bar, v in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(v), ha='center', fontweight='bold', fontsize=11)
ax1.set_title("Distribuição das Classes no Dataset", fontsize=13, fontweight='bold')
ax1.set_ylabel("Quantidade de amostras")
ax1.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig("fig1_distribuicao.png", dpi=150, bbox_inches='tight')
plt.close()

# ── G2: Matriz de Confusão ────────────────────
fig2, ax2 = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Sem doença", "Com doença"])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title("Matriz de Confusão", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig2_confusao.png", dpi=150, bbox_inches='tight')
plt.close()

# ── G3: Importância das Features ─────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
colors_feat = [AZUL if i < 5 else CINZA for i in range(len(idx))]
ax3.bar(range(len(idx)),
        importances[idx], color=colors_feat, edgecolor='white')
ax3.set_xticks(range(len(idx)))
ax3.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
ax3.set_title("Importância das Features (Árvore de Decisão)", fontsize=13, fontweight='bold')
ax3.set_ylabel("Importância")
patch1 = mpatches.Patch(color=AZUL,  label='Top 5 mais importantes')
patch2 = mpatches.Patch(color=CINZA, label='Demais features')
ax3.legend(handles=[patch1, patch2])
plt.tight_layout()
plt.savefig("fig3_importancia.png", dpi=150, bbox_inches='tight')
plt.close()

# ── G4: Acurácia por profundidade ─────────────
depths = range(1, 11)
train_acc, test_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, m.predict(X_train)))
    test_acc.append(accuracy_score(y_test,  m.predict(X_test)))

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(depths, train_acc, 'o-', color=AZUL,  label='Treino',  linewidth=2)
ax4.plot(depths, test_acc,  's-', color=VERM,  label='Teste',   linewidth=2)
ax4.axvline(x=4, color=VERDE, linestyle='--', label='Profundidade escolhida (4)')
ax4.set_xlabel("Profundidade Máxima")
ax4.set_ylabel("Acurácia")
ax4.set_title("Acurácia vs Profundidade da Árvore", fontsize=13, fontweight='bold')
ax4.legend()
ax4.set_ylim(0.5, 1.02)
plt.tight_layout()
plt.savefig("fig4_profundidade.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n=== Gráficos salvos ===")

# Salvar métricas para o relatório
results = {
    'acc': acc,
    'report': report,
    'cm': cm,
    'importances': importances,
    'feature_names': feature_names,
    'idx': idx,
    'n_train': X_train.shape[0],
    'n_test': X_test.shape[0],
    'n_total': df.shape[0],
}

import json
with open("results.json", "w") as f:
    json.dump({
        'acc': float(acc),
        'n_train': int(results['n_train']),
        'n_test':  int(results['n_test']),
        'n_total': int(results['n_total']),
        'cm': cm.tolist(),
        'top_features': [feature_names[i] for i in idx[:5]],
    }, f)

print("Resultados salvos em results.json")
