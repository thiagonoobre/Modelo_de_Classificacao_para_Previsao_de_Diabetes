# 🧪 Projeto de Diagnóstico de Diabetes com Regressão Logística

![Regressão Logística](https://img.shields.io/badge/Modelo-Regressão%20Logística-blueviolet)
![Python](https://img.shields.io/badge/Linguagem-Python-blue)
![Bibliotecas](https://img.shields.io/badge/Bibliotecas-Pandas%2C%20Matplotlib%2C%20Seaborn%2C%20Scikit--learn-brightgreen)
![Status](https://img.shields.io/badge/Status-Concluído-success)

## 📝 Visão Geral do Projeto


Este projeto tem como objetivo principal desenvolver um modelo de classificação utilizando **Regressão Logística** para prever a probabilidade de um indivíduo desenvolver diabetes, com base em dados de saúde. A detecção precoce de diabetes é crucial para a gestão eficaz da doença e a prevenção de complicações graves. O desenvolvimento deste modelo visa fornecer uma ferramenta auxiliar para o diagnóstico, contribuindo para intervenções médicas mais rápidas e personalizadas.

O notebook detalha todas as etapas: desde a análise exploratória e tratamento de dados, até a avaliação aprofundada do desempenho do modelo.

## 📊 Dataset Utilizado

O conjunto de dados utilizado é o `Pima Indians Diabetes Database`, disponível no Kaggle. Ele contém diversas variáveis diagnósticas e uma variável de resultado (`Outcome`) indicando a presença ou ausência de diabetes.

**Fonte**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

**Variáveis Chave:**
* `Pregnancies`: Número de vezes grávida
* `Glucose`: Concentração de glicose no plasma a 2 horas em um teste oral de tolerância à glicose
* `BloodPressure`: Pressão diastólica (mm Hg)
* `SkinThickness`: Espessura da dobra da pele do tríceps (mm)
* `Insulin`: Insulina sérica de 2 horas (mu U/ml)
* `BMI`: Índice de Massa Corporal (peso em kg / (altura em m)^2)
* `DiabetesPedigreeFunction`: Função de pedigree de diabetes (uma função que pontua o risco de diabetes com base no histórico familiar)
* `Age`: Idade (anos)
* `Outcome`: Variável de classe (0 para não diabético, 1 para diabético)

## ✨ Principais Etapas e Análises

1.  **Importação de Dados e Bibliotecas**: Carregamento do dataset e das bibliotecas essenciais para análise.
2.  **Análise Exploratória de Dados (EDA)**:
    * Verificação da estrutura do dataset (`.info()`, `.describe()`).
    * **Tratamento de Valores Biologicamente Implausíveis**:
        * Identificação e tratamento de valores zero (`0`) em colunas como `Glucose`, `BloodPressure` e `BMI`, que são biologicamente inviáveis. Esses zeros foram imputados pela **mediana** da respectiva coluna para manter a integridade dos dados e minimizar o impacto de *outliers*.
        * Para as colunas `SkinThickness` e `Insulin`, onde valores zero também são biologicamente implausíveis, uma estratégia mais avançada foi aplicada: os zeros foram substituídos por valores `NaN` e, posteriormente, imputados utilizando o **`KNNImputer`** (com `n_neighbors=5`). Essa técnica preenche os valores ausentes com base na média dos vizinhos mais próximos, considerando a similaridade entre as instâncias e preservando melhor as relações nos dados.
    * Análise da distribuição das variáveis e correlações.
    * **Verificação de Multicolinearidade:**
        * Realizada uma análise de **Variance Inflation Factor (VIF)** para as variáveis preditoras. O objetivo foi identificar e mitigar a multicolinearidade, um fenômeno que pode impactar a estabilidade e a interpretabilidade dos coeficientes do modelo.
        * **Resultado do VIF:** Os valores de VIF indicaram baixa multicolinearidade entre as features (todos abaixo de 2), confirmando que não há problemas graves que comprometam a modelagem.
3.  **Pré-processamento de Dados**:
    * Divisão do dataset em conjuntos de treino e teste para garantir uma avaliação imparcial do modelo.
    * Escalamento das features utilizando **`StandardScaler`** para padronizar as variáveis (média zero e desvio padrão um). Este passo é crucial para algoritmos baseados em distância e para a interpretação de coeficientes em modelos lineares.
4.  **Modelagem e Treinamento**:
    * Aplicação do modelo de **Regressão Logística**, uma técnica robusta e interpretável para problemas de classificação binária.
    * Treinamento do modelo com os dados de treino, utilizando um pipeline (`make_pipeline`) para encapsular o escalamento e o treinamento, prevenindo assim o *data leakage*.
5.  **Avaliação do Modelo**:
    * Geração da Matriz de Confusão para entender os acertos e erros do classificador (Verdadeiros Positivos, Falsos Negativos, etc.).
    * Cálculo de métricas essenciais como Acurácia, Precisão, Recall e F1-Score para ambas as classes.
    * Análise da Curva ROC e AUC para avaliar a capacidade discriminatória geral do modelo em diferentes limiares.
    * Avaliação da Curva de Precisão-Recall (PR Curve) para compreender o trade-off entre Precisão e Recall, crucial para classes desbalanceadas.
    * Cálculo do estatístico KS (Kolmogorov-Smirnov) para medir a separabilidade das classes.
    * **Análise de Interpretabilidade:** Avaliação dos coeficientes do modelo para entender a contribuição e a direção do impacto de cada feature nas previsões de diabetes.

## 📈 Resultados e Análise

Para avaliar o desempenho do modelo de Regressão Logística, foram analisadas diversas métricas, considerando a sensibilidade do problema de diagnóstico de diabetes.

Inicialmente, com um **limiar (threshold) padrão de 0.5**, a **Matriz de Confusão** do modelo apresentou o seguinte cenário:

![](https://raw.githubusercontent.com/thiagonoobre/Modelo_de_Classificacao_para_Previsao_de_Diabetes/refs/heads/main/imagens/Matriz_Confus%C3%A3o_Threshold05.png)

Nesta configuração, obtivemos **83 Verdadeiros Positivos (TP)**, que representam diagnósticos corretos de diabetes. No entanto, foram registrados **16 Falsos Positivos (FP)** (alarmes falsos, onde o modelo previu diabetes, mas o paciente era saudável) e, mais criticamente, **21 Falsos Negativos (FN)** (pacientes com diabetes não diagnosticados, um erro de alta gravidade em saúde).

Dada a importância de minimizar os Falsos Negativos em contextos médicos, o limiar de decisão do modelo foi ajustado para **0.2**. Essa alteração resultou em um novo balanço, visível na **Matriz de Confusão** a seguir:

![](https://raw.githubusercontent.com/thiagonoobre/Modelo_de_Classificacao_para_Previsao_de_Diabetes/refs/heads/main/imagens/Matriz_Confus%C3%A3o_Threshold02.png)

Com o limiar em 0.2, conseguimos **reduzir significativamente os Falsos Negativos (FN) de 21 para apenas 6 casos**, um avanço crucial na detecção de diabetes. Contudo, essa melhoria veio com um *trade-off*: os **Falsos Positivos (FP) aumentaram de 16 para 47 casos**.

As métricas consolidadas do modelo, operando com o limiar de 0.2, são apresentadas abaixo:

| Métrica | Valor |
| :---------------------- | :---- |
| Acurácia | 0.66 |
| Recall (Sensibilidade) | **0.89** |
| Precisão | 0.51 |
| PR AUC | 0.71 |
| AUC (ROC) | 0.82 |
| KS Statistic | 0.51 |

**Análise das Métricas Chave:**

  * A **alta Sensibilidade (Recall) de 0.89** para a classe 'diabetes' confirma o sucesso na estratégia de minimizar Falsos Negativos, garantindo que a maioria dos casos reais seja identificada.
  * A **Precisão de 0.51** para a classe 'diabetes' reflete o aumento nos Falsos Positivos, indicando que cerca de metade das previsões de 'diabetes' são, de fato, corretas.
  * O **AUC (ROC) de 0.82** demonstra uma boa capacidade discriminatória geral do modelo, sugerindo que ele é eficaz em distinguir entre classes positiva e negativa em diversos limiares.
  * O **PR AUC de 0.71** é um indicador robusto para problemas com classes desbalanceadas, confirmando um desempenho aceitável no balanço entre Precisão e Recall.
  * O **Estatístico KS de 0.51** indica uma separação moderada a boa entre as distribuições de probabilidade das classes.

As variáveis mais importantes para a predição, determinadas pela magnitude de seus coeficientes, são:

1.  **Glucose** (Coeficiente: 1.3948)
2.  **BMI** (Coeficiente: 0.8449)
3.  **Age** (Coeficiente: 0.5622)

### Conclusão

O modelo de Regressão Logística desenvolvido demonstra um desempenho geral robusto, priorizando com sucesso a detecção de casos de diabetes. Apesar do aumento nos Falsos Positivos e uma acurácia geral de 0.66, o foco estratégico na minimização dos Falsos Negativos está alinhado com as necessidades críticas de um diagnóstico precoce em saúde. As métricas e a interpretabilidade do modelo sugerem que ele serve como um sólido ponto de partida para apoiar decisões clínicas, com potencial para refinamentos futuros.

## 🚀 Próximos Passos

Para continuar aprimorando e validando o modelo de diagnóstico de diabetes, os próximos passos sugeridos incluem:

1.  **Análise Qualitativa dos Erros**: Investigar Falso Positivos e Falsos Negativos para entender as características comuns desses pacientes.
2.  **Otimização de Hiperparâmetros**: Ajustar parâmetros do modelo usando técnicas como Grid Search ou Random Search com validação cruzada.
3.  **Exploração de Outros Modelos**: Testar algoritmos de classificação alternativos (e.g., Random Forest, XGBoost) para comparar o desempenho e o potencial de melhoria.

## 📁 Execução

Clone este repositório e execute o notebook:

```bash
git clone https://github.com/thiagonoobre/Modelo_de_Classificacao_para_Previsao_de_Diabetes.git
cd Modelo_de_Classificacao_para_Previsao_de_Diabetes
jupyter notebook
```
