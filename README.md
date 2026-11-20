# 📉 Estudos — *Estatística Prática para Cientistas de Dados* (Peter Bruce & Andrew Bruce)

> **A melhor forma de estudar um livro técnico é ler e praticar ao mesmo tempo.**
> Leia uma seção, feche o livro e escreva o código com as próprias mãos. Só depois confira o gabarito.

Este repositório registra meu estudo de *Estatística Prática para Cientistas de Dados*, que explica os conceitos de estatística mais úteis para quem trabalha com dados.
Cada capítulo virou um caderno Jupyter. Cada seção aponta para a página do PDF e, quando cabe, traz um exercício original em Python.

## 🔁 O método

| Passo | O que fazer | Onde |
|---|---|---|
| 1. Ler | Ler a seção indicada | PDF do livro (página no caderno) |
| 2. Praticar | Escrever a solução do 🧪 exercício **sem olhar a resposta** | `praticar/capXX_*.ipynb` |
| 3. Conferir | Comparar com o gabarito e entender a diferença | `resolvidos/capXX_*.ipynb` |
| 4. Resumir | Explicar com as próprias palavras nas células 📝 | `praticar/` |
| 5. Registrar | Commit + push | GitHub |

## 📚 Conteúdo (65 exercícios em Python)

| Cap. | Tema | Exemplos |
|---|---|---|
| 1 | Análise exploratória | estimativas robustas, MAD, percentis, boxplot, KDE, valor esperado, correlação, hexbin, tabela de contingência, violino |
| 2 | Distribuições e amostras | viés de sobrevivência, regressão à média, limite central, erro-padrão, *bootstrap*, intervalos, QQ, caudas longas, t, binomial, Poisson, exponencial, Weibull |
| 3 | Experimentos e significância | permutação, erros tipo 1/2 por simulação, Welch, Bonferroni × BH, ANOVA de 1 e 2 fatores, qui-quadrado por reamostragem, Fisher, bandido epsilon-greedy, potência |
| 4 | Regressão | OLS, validação cruzada, *forward* por AIC, intervalo de confiança × previsão, *dummies*, VIF, interações, distância de Cook, splines |
| 5 | Classificação | Naive Bayes, LDA, razão de chances, matriz de confusão, ROC/AUC, *lift*, classe rara |
| 6 | Aprendizado estatístico | KNN e padronização, escolha de k, Gini × entropia, floresta, importância por permutação, *boosting*, busca de hiperparâmetros |
| 7 | Não supervisionado | PCA à mão, silhueta, dendrograma, mistura de normais com BIC, variável dominante |

## 🛠️ Melhorias em relação ao livro

- A edição traduzida traz o código em **R**. Aqui, cada conceito é praticado em **Python** atual: pandas, SciPy, statsmodels 0.14+ e scikit-learn 1.x.
- Não há download de dados: tudo é simulado (quando se conhece a resposta, dá para conferir o método) ou vem das bases do scikit-learn.
- Os testes clássicos são comparados com a versão por reamostragem (permutação, *bootstrap*, qui-quadrado simulado), como o livro recomenda.
- O gabarito inteiro é executado por `ferramentas/testar_cadernos.py`.
- Código oficial da 2ª edição (R + Python, GPL-3.0): [gedeck/practical-statistics-for-data-scientists](https://github.com/gedeck/practical-statistics-for-data-scientists).

## ▶️ Como usar

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-estudos.txt
.venv\Scripts\python.exe ferramentas\testar_cadernos.py   # opcional: roda o gabarito
```

Abra `praticar/cap01_*.ipynb` no VS Code e escolha o kernel `.venv`. O [Guia de Estudos (PDF)](Guia_de_Estudos.pdf) explica o passo a passo.

## ⚖️ Direitos autorais

O livro **não** está neste repositório. Para estudar, use a sua própria cópia.
Os exercícios, os cadernos, os scripts e o guia são autorais e estão sob licença [MIT](LICENSE).
