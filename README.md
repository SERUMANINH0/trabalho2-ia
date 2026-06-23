# Trabalho 2 — Inteligência Artificial

Classificação de doenças cardíacas usando Árvore de Decisão.

**Curso:** Engenharia de Software — FAG Cascavel  
**Aluno:** Venery Gutiery Porto Rufino  
**Professor:** Roberval Requião Junior  

---

## O que esse código faz

Treina um modelo de Machine Learning para prever se um paciente tem ou não indicativo de doença cardíaca, com base em 13 dados clínicos como idade, colesterol, pressão arterial e frequência cardíaca.

O algoritmo usado foi Árvore de Decisão (`DecisionTreeClassifier` do scikit-learn). O modelo atingiu **83,61% de acurácia** no conjunto de teste.

## Dataset

Baseado no Heart Disease Dataset do UCI (Cleveland). Os dados foram gerados de forma sintética com numpy (semente 42) respeitando os mesmos intervalos e correlações do dataset original, para garantir reprodutibilidade.

- 303 pacientes
- 13 atributos de entrada
- Target: 0 = sem doença, 1 = com doença

## Como rodar

Instale as dependências:

```bash
pip3 install pandas numpy scikit-learn matplotlib seaborn
```

Execute o código:

```bash
python3 heart_disease_dt.py
```

Os gráficos são salvos automaticamente na pasta do projeto:

- `fig1_distribuicao.png`
- `fig2_confusao.png`
- `fig3_importancia.png`
- `fig4_profundidade.png`

## Resultados

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Sem doença | 0.64 | 0.64 | 0.64 |
| Com doença | 0.89 | 0.89 | 0.89 |
| **Acurácia geral** | | | **0.84** |

## Arquivos

```
trabalho2-ia/
├── heart_disease_dt.py       # código principal
├── relatorio_trabalho2_v2.pdf  # relatório completo
├── fig1_distribuicao.png
├── fig2_confusao.png
├── fig3_importancia.png
├── fig4_profundidade.png
└── results.json              # métricas salvas
```