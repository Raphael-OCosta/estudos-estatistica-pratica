"""Exercícios práticos em Python criados para estudar o livro, um ou mais por seção.

Formato: (capítulo, início do título da seção, nome, enunciado em Markdown, solução em Python).
Os dados são simulados ou vêm das bases que acompanham o scikit-learn (diabetes, câncer de mama, vinhos),
então nada precisa ser baixado. Os exemplos são originais (não copiados do livro).
"""

EXERCICIOS = [
    # ---------------- 1. Análise exploratória ----------------
    (1, "Estimativas de Localização", "Média, mediana, aparada e ponderada",
     "Gere as rendas de 1.000 pessoas com `rng.lognormal(8, 1)`. Calcule a média, a mediana, a média aparada em 10% "
     "(`stats.trim_mean`) e a média ponderada usando pesos aleatórios. Por que a média fica tão acima da mediana?",
     '''renda = rng.lognormal(8, 1, 1000)
pesos = rng.uniform(0.5, 2, 1000)
print(f"média {renda.mean():.0f} | mediana {np.median(renda):.0f} | aparada {stats.trim_mean(renda, 0.1):.0f} | ponderada {np.average(renda, weights=pesos):.0f}")
# a cauda direita longa (poucos muito ricos) puxa a média para cima'''),
    (1, "Mediana e Estimativas Robustas", "Robustez a um outlier",
     "Pegue 20 salários em torno de 5.000, calcule média e mediana, depois troque um deles por 1.000.000. "
     "Quanto cada estimativa muda?",
     '''s = rng.normal(5000, 500, 20)
antes = (s.mean(), np.median(s)); s[0] = 1_000_000
print([round(x) for x in antes], [round(x) for x in (s.mean(), np.median(s))])'''),
    (1, "Desvio-padrão", "Desvio-padrão × MAD",
     "Para as rendas lognormais, compare desvio-padrão, desvio absoluto médio e MAD (desvio absoluto mediano, "
     "`stats.median_abs_deviation`). Qual é menos afetado pela cauda?",
     '''renda = rng.lognormal(8, 1, 1000)
print(f"dp {renda.std(ddof=1):.0f} | desvio abs. médio {np.mean(np.abs(renda - renda.mean())):.0f} | MAD {stats.median_abs_deviation(renda):.0f}")'''),
    (1, "Estimativas Baseadas em Percentis", "Percentis e amplitude interquartil",
     "Para os dados de diabetes do scikit-learn (`load_diabetes(as_frame=True, scaled=False)`), calcule os percentis "
     "5, 25, 50, 75 e 95 da variável `bmi` e a amplitude interquartil (IQR).",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
p = np.percentile(df["bmi"], [5, 25, 50, 75, 95]); print(p, "IQR =", p[3] - p[1])'''),
    (1, "Percentis e Boxplots", "Boxplot por grupo",
     "Faça um boxplot do `target` (progressão da doença) por sexo (`sex`) com pandas (`df.boxplot`).",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
df.boxplot(column="target", by="sex"); plt.suptitle(""); plt.show()'''),
    (1, "Tabela de Frequências", "Tabela de frequências e histograma",
     "Divida `bmi` em 8 faixas de mesmo tamanho com `pd.cut`, mostre a contagem de cada faixa (`value_counts`, "
     "ordenada pela faixa) e o histograma correspondente.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
print(pd.cut(df["bmi"], 8).value_counts().sort_index())
df["bmi"].plot.hist(bins=8, edgecolor="black"); plt.show()'''),
    (1, "Estimativas de Densidade", "Densidade (KDE) sobre o histograma",
     "Desenhe o histograma normalizado (`density=True`) do `target` e, por cima, a densidade estimada com "
     "`stats.gaussian_kde`.",
     '''from sklearn.datasets import load_diabetes
y = load_diabetes(as_frame=True, scaled=False).frame["target"]
plt.hist(y, bins=25, density=True, alpha=0.5)
xs = np.linspace(y.min(), y.max(), 200); plt.plot(xs, stats.gaussian_kde(y)(xs), lw=2); plt.show()'''),
    (1, "Explorando Dados Binários", "Proporções e gráfico de barras",
     "Simule 500 atrasos de voo com causas `[\"companhia\", \"clima\", \"tráfego\", \"segurança\"]` e probabilidades "
     "`[0.4, 0.15, 0.4, 0.05]`. Mostre a proporção de cada causa e um gráfico de barras.",
     '''causas = pd.Series(rng.choice(["companhia", "clima", "tráfego", "segurança"], 500, p=[0.4, 0.15, 0.4, 0.05]))
prop = causas.value_counts(normalize=True); print(prop.round(3))
prop.plot.bar(); plt.show()'''),
    (1, "Valor Esperado", "Valor esperado de uma oferta",
     "Um serviço tem 5% dos clientes no plano de R$ 300/mês, 15% no de R$ 50 e 80% gratuitos. Calcule o valor "
     "esperado por cliente e confira com 100.000 clientes simulados.",
     '''valores, probs = np.array([300, 50, 0]), np.array([0.05, 0.15, 0.80])
print((valores * probs).sum(), rng.choice(valores, 100_000, p=probs).mean().round(2))'''),
    (1, "Correlação", "Matriz de correlação",
     "Para as variáveis `age, bmi, bp, s1, s5, target` dos dados de diabetes, mostre a matriz de correlação de "
     "Pearson e a de Spearman e desenhe a de Pearson com `plt.imshow`.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame[["age", "bmi", "bp", "s1", "s5", "target"]]
c = df.corr(); print(c.round(2)); print(df.corr(method="spearman").round(2))
plt.imshow(c, cmap="RdBu", vmin=-1, vmax=1); plt.xticks(range(6), c.columns); plt.yticks(range(6), c.columns); plt.colorbar(); plt.show()'''),
    (1, "Compartimentação Hexagonal", "Hexbin para muitos pontos",
     "Gere 50.000 pares (x normal; y = 2x + ruído) e compare um `scatter` com um `hexbin` (`gridsize=40`).",
     '''x = rng.normal(size=50_000); y = 2 * x + rng.normal(size=50_000)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(x, y, s=1, alpha=0.2); ax[1].hexbin(x, y, gridsize=40, cmap="Blues"); plt.show()'''),
    (1, "Duas Variáveis Categóricas", "Tabela de contingência",
     "Simule 1.000 empréstimos com nota (A–D) e situação (pago, atrasado, inadimplente), onde notas piores têm mais "
     "inadimplência. Monte a tabela cruzada com proporções por linha (`pd.crosstab(..., normalize=\"index\")`).",
     '''nota = rng.choice(list("ABCD"), 1000)
p_inad = pd.Series(nota).map({"A": 0.03, "B": 0.08, "C": 0.15, "D": 0.3}).to_numpy()
u = rng.random(1000)
sit = np.where(u < p_inad, "inadimplente", np.where(u < p_inad + 0.1, "atrasado", "pago"))
print(pd.crosstab(nota, sit, normalize="index").round(3))'''),
    (1, "Dados Categóricos e Numéricos", "Violino por categoria",
     "Com os dados de vinhos (`load_wine(as_frame=True)`), desenhe o teor alcoólico (`alcohol`) por classe de vinho "
     "com `plt.violinplot`.",
     '''from sklearn.datasets import load_wine
df = load_wine(as_frame=True).frame
grupos = [g["alcohol"].to_numpy() for _, g in df.groupby("target")]
plt.violinplot(grupos, showmedians=True); plt.xticks([1, 2, 3], ["classe 0", "classe 1", "classe 2"]); plt.show()'''),

    # ---------------- 2. Distribuições e amostras ----------------
    (2, "Viés de Seleção", "Viés de sobrevivência",
     "Simule 1.000 fundos com retorno anual normal(0, 10%). Fique só com os \"sobreviventes\" (retorno > −5%) e "
     "compare a média dos sobreviventes com a de todos.",
     '''ret = rng.normal(0, 0.10, 1000)
print(f"todos {ret.mean():.3%} | sobreviventes {ret[ret > -0.05].mean():.3%}")'''),
    (2, "Regressão à Média", "Regressão à média",
     "Cada jogador tem uma habilidade fixa e, em cada temporada, desempenho = habilidade + sorte. Simule 2 temporadas "
     "para 1.000 jogadores, pegue os 10% melhores da 1ª e veja a média deles na 2ª.",
     '''hab = rng.normal(0, 1, 1000)
t1 = hab + rng.normal(0, 1, 1000); t2 = hab + rng.normal(0, 1, 1000)
top = t1 >= np.quantile(t1, 0.9)
print(round(t1[top].mean(), 2), round(t2[top].mean(), 2))  # caem na 2ª temporada'''),
    (2, "Teorema de Limite Central", "Distribuição amostral da média",
     "Partindo de uma exponencial (bem assimétrica), tire 2.000 amostras de tamanho 5, 20 e 100 e desenhe o "
     "histograma da média amostral de cada caso.",
     '''fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for a, n in zip(ax, [5, 20, 100]):
    a.hist(rng.exponential(1, (2000, n)).mean(axis=1), bins=40); a.set_title(f"n = {n}")
plt.show()'''),
    (2, "Erro-padrão", "Erro-padrão: fórmula × simulação",
     "Para amostras de tamanho 50 de uma normal(10, 3), compare o erro-padrão da média pela fórmula s/√n com o "
     "desvio-padrão de 5.000 médias simuladas.",
     '''amostra = rng.normal(10, 3, 50)
print(round(amostra.std(ddof=1) / np.sqrt(50), 3), round(rng.normal(10, 3, (5000, 50)).mean(axis=1).std(), 3))'''),
    (2, "O Bootstrap", "Bootstrap da mediana",
     "Com 200 rendas lognormais, estime o erro-padrão e o viés da mediana por *bootstrap* (2.000 reamostragens).",
     '''renda = rng.lognormal(8, 1, 200)
boot = np.array([np.median(rng.choice(renda, renda.size)) for _ in range(2000)])
print(f"mediana {np.median(renda):.0f} | viés {boot.mean() - np.median(renda):.1f} | erro-padrão {boot.std(ddof=1):.1f}")'''),
    (2, "Intervalos de Confiança", "Intervalo bootstrap × intervalo t",
     "Para 40 valores normais(100, 15), calcule o intervalo de 95% da média pelo bootstrap (percentis 2,5 e 97,5) "
     "e pela fórmula t (`stats.t.interval`).",
     '''x = rng.normal(100, 15, 40)
boot = [rng.choice(x, x.size).mean() for _ in range(5000)]
print(np.percentile(boot, [2.5, 97.5]).round(2), np.round(stats.t.interval(0.95, x.size - 1, x.mean(), stats.sem(x)), 2))'''),
    (2, "Normal Padrão e Gráficos QQ", "Gráfico QQ",
     "Faça gráficos QQ (`stats.probplot(..., plot=ax)`) para 300 valores normais e 300 valores lognormais. Como a "
     "assimetria aparece no gráfico?",
     '''fig, ax = plt.subplots(1, 2, figsize=(10, 4))
stats.probplot(rng.normal(size=300), plot=ax[0]); ax[0].set_title("normal")
stats.probplot(rng.lognormal(size=300), plot=ax[1]); ax[1].set_title("lognormal"); plt.show()'''),
    (2, "Distribuições de Cauda Longa", "Cauda longa: eventos extremos",
     "Compare a probabilidade de um valor passar de 4 desvios-padrão numa normal e numa t de Student com 3 graus de "
     "liberdade (padronizada pela sua variância). Quantas vezes é maior?",
     '''p_norm = 2 * stats.norm.sf(4)
dp_t = np.sqrt(3 / (3 - 2))
p_t = 2 * stats.t.sf(4 * dp_t, 3)
print(f"{p_norm:.2e} {p_t:.2e} -> {p_t / p_norm:.0f}x")'''),
    (2, "Distribuição t de Student", "t × normal",
     "Desenhe a densidade da normal padrão e da t com 2, 5 e 30 graus de liberdade. Para qual g.l. elas ficam quase "
     "iguais?",
     '''xs = np.linspace(-4, 4, 300)
plt.plot(xs, stats.norm.pdf(xs), "k", lw=2, label="normal")
for gl in [2, 5, 30]: plt.plot(xs, stats.t.pdf(xs, gl), label=f"t({gl})")
plt.legend(); plt.show()'''),
    (2, "Distribuição Binomial", "Binomial: cliques em anúncios",
     "Um anúncio tem 2% de cliques. Em 200 exibições, qual é a chance de 0 cliques, de exatamente 4 e de 8 ou mais? "
     "Confira a média e a variância por simulação.",
     '''b = stats.binom(200, 0.02)
print(round(b.pmf(0), 4), round(b.pmf(4), 4), round(b.sf(7), 4))
sim = rng.binomial(200, 0.02, 100_000); print(sim.mean(), round(sim.var(), 3), b.mean(), b.var())'''),
    (2, "Distribuições Poisson", "Poisson: chamadas por minuto",
     "Uma central recebe em média 2 chamadas por minuto. Qual é a probabilidade de 5 ou mais chamadas num minuto? "
     "Quantas chamadas cobrem 99% dos minutos?",
     '''p = stats.poisson(2); print(round(p.sf(4), 4), p.ppf(0.99))'''),
    (2, "Distribuição Exponencial", "Tempo entre chamadas",
     "Com a mesma taxa (2 por minuto), simule 10.000 tempos entre chamadas (exponencial) e estime P(espera > 1 min). "
     "Compare com e⁻².",
     '''t = rng.exponential(1 / 2, 10_000); print(round((t > 1).mean(), 4), round(np.exp(-2), 4))'''),
    (2, "Distribuição Weibull", "Weibull e taxa de falha",
     "Simule 5.000 tempos de vida de peças com Weibull de forma 1,5 e escala 1.000 h e ajuste com "
     "`stats.weibull_min.fit(..., floc=0)`. Os parâmetros voltam?",
     '''vida = stats.weibull_min.rvs(1.5, scale=1000, size=5000, random_state=1)
forma, loc, escala = stats.weibull_min.fit(vida, floc=0); print(round(forma, 3), round(escala, 1))'''),

    # ---------------- 3. Experimentos e significância ----------------
    (3, "Teste de Permutação", "Teste de permutação",
     "Tempo na página: A tem 30 sessões ~ normal(60, 20) e B tem 25 ~ normal(75, 20). Faça um teste de permutação "
     "com 5.000 embaralhamentos para a diferença de médias e calcule o p-valor bilateral.",
     '''a = rng.normal(60, 20, 30); b = rng.normal(75, 20, 25)
obs = b.mean() - a.mean(); tudo = np.concatenate([a, b])
difs = []
for _ in range(5000):
    p = rng.permutation(tudo); difs.append(p[30:].mean() - p[:30].mean())
print(round(obs, 2), np.mean(np.abs(difs) >= abs(obs)))'''),
    (3, "Erros Tipo 1", "Erro tipo 1 e tipo 2 por simulação",
     "Com α = 5% e teste t para duas amostras de 20: (a) sem efeito real, qual fração dos 2.000 testes rejeita H₀? "
     "(b) com diferença real de 0,5 desvio, qual fração não rejeita (erro tipo 2)?",
     '''def rejeita(efeito):
    return stats.ttest_ind(rng.normal(0, 1, 20), rng.normal(efeito, 1, 20)).pvalue < 0.05
print(np.mean([rejeita(0) for _ in range(2000)]), 1 - np.mean([rejeita(0.5) for _ in range(2000)]))'''),
    (3, "Testes t", "Teste t de Welch",
     "Compare a pressão (`bp`) de homens e mulheres nos dados de diabetes com `stats.ttest_ind(..., equal_var=False)`.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
g1, g2 = [g["bp"] for _, g in df.groupby("sex")]
print(stats.ttest_ind(g1, g2, equal_var=False))'''),
    (3, "Testagem Múltipla", "Correção para testes múltiplos",
     "Gere 100 p-valores de testes sem efeito (uniformes) e 5 de efeitos reais (`rng.uniform(0, 0.001, 5)`). Conte "
     "quantos passam em α = 5% sem correção, com Bonferroni e com Benjamini-Hochberg "
     "(`statsmodels.stats.multitest.multipletests`).",
     '''from statsmodels.stats.multitest import multipletests
p = np.concatenate([rng.uniform(0, 1, 100), rng.uniform(0, 0.001, 5)])
print((p < 0.05).sum(), multipletests(p, 0.05, "bonferroni")[0].sum(), multipletests(p, 0.05, "fdr_bh")[0].sum())'''),
    (3, "Estatística F", "ANOVA de um fator",
     "Simule o tempo de permanência em 4 versões de página (médias 60, 62, 61 e 70, desvio 10, 25 sessões cada) e "
     "rode `stats.f_oneway`. Depois faça a mesma ANOVA com `smf.ols` + `sm.stats.anova_lm`.",
     '''medias = [60, 62, 61, 70]
df = pd.DataFrame([(f"p{i}", t) for i, m in enumerate(medias) for t in rng.normal(m, 10, 25)], columns=["pagina", "tempo"])
print(stats.f_oneway(*[g["tempo"] for _, g in df.groupby("pagina")]))
print(sm.stats.anova_lm(smf.ols("tempo ~ C(pagina)", df).fit()))'''),
    (3, "ANOVA Bidirecional", "ANOVA com dois fatores",
     "Acrescente um segundo fator, `dispositivo` (celular/computador, +5 s no computador), e ajuste "
     "`tempo ~ C(pagina) * C(dispositivo)`. Mostre a tabela ANOVA.",
     '''linhas = []
for i, m in enumerate([60, 62, 61, 70]):
    for disp, extra in [("celular", 0), ("computador", 5)]:
        linhas += [(f"p{i}", disp, t) for t in rng.normal(m + extra, 10, 15)]
df = pd.DataFrame(linhas, columns=["pagina", "dispositivo", "tempo"])
print(sm.stats.anova_lm(smf.ols("tempo ~ C(pagina) * C(dispositivo)", df).fit(), typ=2).round(3))'''),
    (3, "Teste de Qui Quadrado: Uma", "Qui-quadrado por reamostragem e pela teoria",
     "Três manchetes tiveram cliques/visualizações 14/1000, 8/1000 e 12/1000. Faça o teste qui-quadrado com "
     "`stats.chi2_contingency` e por reamostragem (5.000 simulações sob H₀).",
     '''cliques = np.array([14, 8, 12]); n = 1000
tab = np.array([cliques, n - cliques])
chi2, p, *_ = stats.chi2_contingency(tab); print(round(chi2, 3), round(p, 4))
taxa = cliques.sum() / (3 * n)
def chi2_de(c):
    esp = np.array([[taxa * n] * 3, [(1 - taxa) * n] * 3]); obs = np.array([c, n - c])
    return ((obs - esp) ** 2 / esp).sum()
obs = chi2_de(cliques)
sims = [chi2_de(rng.binomial(n, taxa, 3)) for _ in range(5000)]
print(round(obs, 3), np.mean(np.array(sims) >= obs))'''),
    (3, "Teste Exato de Fisher", "Fisher para amostras pequenas",
     "Num teste com poucos dados, A converteu 1 de 10 e B 7 de 12. Rode o teste exato de Fisher e compare com o "
     "qui-quadrado (que não é confiável aqui).",
     '''tab = [[1, 9], [7, 5]]
print(stats.fisher_exact(tab), stats.chi2_contingency(tab)[1])'''),
    (3, "Algoritmo de Bandido", "Bandido de vários braços (epsilon-greedy)",
     "Três ofertas convertem 4%, 5% e 7% (você não sabe). Simule 10.000 visitantes com epsilon-greedy (ε = 0,1): "
     "explora ao acaso com prob. ε, senão mostra a de maior taxa observada. Quantas vezes cada oferta foi mostrada?",
     '''taxas = np.array([0.04, 0.05, 0.07]); mostra = np.zeros(3); conv = np.zeros(3)
for _ in range(10_000):
    braco = rng.integers(3) if rng.random() < 0.1 or mostra.min() == 0 else int(np.argmax(conv / mostra))
    mostra[braco] += 1; conv[braco] += rng.random() < taxas[braco]
print(mostra.astype(int), (conv / mostra).round(4))'''),
    (3, "Tamanho da Amostra", "Potência e tamanho de amostra",
     "Quantos visitantes por grupo são necessários para detectar uma mudança de conversão de 10% para 11% com "
     "α = 5% e potência de 80%? Use `NormalIndPower` e `proportion_effectsize` do statsmodels.",
     '''from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
ef = proportion_effectsize(0.11, 0.10)
print(round(NormalIndPower().solve_power(ef, alpha=0.05, power=0.8)))'''),

    # ---------------- 4. Regressão e previsão ----------------
    (4, "Regressão Linear Simples", "Regressão simples com statsmodels",
     "Com os dados de diabetes, ajuste `target ~ bmi` com `smf.ols`, mostre os coeficientes e o R² e desenhe a reta "
     "sobre os pontos.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
m = smf.ols("target ~ bmi", df).fit(); print(m.params.round(3), round(m.rsquared, 3))
plt.scatter(df["bmi"], df["target"], s=8); xs = np.linspace(df["bmi"].min(), df["bmi"].max(), 2)
plt.plot(xs, m.params["Intercept"] + m.params["bmi"] * xs, "r"); plt.show()'''),
    (4, "Regressão Linear Múltipla", "Regressão múltipla e RMSE",
     "Ajuste `target ~ age + sex + bmi + bp + s1 + s5`, mostre o `summary()` e calcule o RMSE dos resíduos.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
m = smf.ols("target ~ age + sex + bmi + bp + s1 + s5", df).fit()
print(m.summary().tables[1]); print("RMSE:", round(np.sqrt(np.mean(m.resid ** 2)), 2))'''),
    (4, "Validação Cruzada", "Validação cruzada com scikit-learn",
     "Compare, com validação cruzada de 5 lotes (RMSE), a regressão só com `bmi` e a regressão com todas as 10 "
     "variáveis.",
     '''from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
X, y = load_diabetes(return_X_y=True, as_frame=True)
for cols in (["bmi"], list(X.columns)):
    s = cross_val_score(LinearRegression(), X[cols], y, cv=5, scoring="neg_root_mean_squared_error")
    print(len(cols), "variáveis -> RMSE", round(-s.mean(), 2))'''),
    (4, "Seleção de Modelo", "Seleção passo a passo por AIC",
     "Implemente uma seleção *forward*: comece com `target ~ 1` e, a cada passo, acrescente a variável que mais reduz "
     "o AIC; pare quando nenhuma reduzir.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
restantes, escolhidas = [c for c in df.columns if c != "target"], []
aic = smf.ols("target ~ 1", df).fit().aic
while restantes:
    cand = {v: smf.ols("target ~ " + " + ".join(escolhidas + [v]), df).fit().aic for v in restantes}
    v, novo = min(cand.items(), key=lambda kv: kv[1])
    if novo >= aic: break
    escolhidas.append(v); restantes.remove(v); aic = novo
print(escolhidas, round(aic, 1))'''),
    (4, "Intervalos de Confiança e Previsão", "Intervalo de confiança × de previsão",
     "No modelo `target ~ bmi`, calcule para bmi = 30 o intervalo de confiança da média e o intervalo de previsão de "
     "um paciente novo (`get_prediction(...).summary_frame(alpha=0.05)`). Por que o segundo é mais largo?",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
m = smf.ols("target ~ bmi", df).fit()
print(m.get_prediction(pd.DataFrame({"bmi": [30]})).summary_frame(alpha=0.05).round(1))'''),
    (4, "Representação de Variáveis Fictícias", "Variáveis dummy",
     "Crie um fator `faixa_idade` (jovem < 40, meia-idade < 60, idoso) e ajuste `target ~ bmi + C(faixa_idade)`. "
     "Mostre as colunas dummy com `pd.get_dummies(..., drop_first=True)` e interprete um coeficiente.",
     '''from sklearn.datasets import load_diabetes
df = load_diabetes(as_frame=True, scaled=False).frame
df["faixa_idade"] = pd.cut(df["age"], [0, 40, 60, 120], labels=["jovem", "meia_idade", "idoso"])
print(pd.get_dummies(df["faixa_idade"], drop_first=True).head())
print(smf.ols("target ~ bmi + C(faixa_idade)", df).fit().params.round(2))'''),
    (4, "Multicolinearidade", "VIF",
     "Calcule o fator de inflação da variância (VIF) das variáveis `s1` a `s6` dos dados de diabetes "
     "(`variance_inflation_factor`). Quais são muito colineares?",
     '''from sklearn.datasets import load_diabetes
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = sm.add_constant(load_diabetes(as_frame=True, scaled=False).frame[["s1", "s2", "s3", "s4", "s5", "s6"]])
print({c: round(variance_inflation_factor(X.values, i), 1) for i, c in enumerate(X.columns) if c != "const"})'''),
    (4, "Interações", "Interação entre variáveis",
     "Simule vendas = 10 + 2·anúncio + 3·promoção + 4·anúncio·promoção + ruído (anúncio contínuo, promoção 0/1). "
     "Ajuste com e sem o termo de interação e compare o R².",
     '''n = 300
d = pd.DataFrame({"anuncio": rng.uniform(0, 10, n), "promocao": rng.integers(0, 2, n)})
d["vendas"] = 10 + 2 * d.anuncio + 3 * d.promocao + 4 * d.anuncio * d.promocao + rng.normal(0, 3, n)
print(round(smf.ols("vendas ~ anuncio + promocao", d).fit().rsquared, 3), round(smf.ols("vendas ~ anuncio * promocao", d).fit().rsquared, 3))'''),
    (4, "Valores Influentes", "Outliers e influência",
     "Gere uma regressão com 50 pontos e acrescente um ponto muito afastado. Calcule a distância de Cook "
     "(`get_influence().cooks_distance`) e mostre qual ponto tem a maior.",
     '''x = rng.uniform(0, 10, 50); y = 2 * x + rng.normal(0, 1, 50)
x = np.append(x, 25); y = np.append(y, 5)
m = sm.OLS(y, sm.add_constant(x)).fit()
cook = m.get_influence().cooks_distance[0]; print(int(np.argmax(cook)), round(cook.max(), 2))'''),
    (4, "Splines", "Polinômio × spline",
     "Para y = sin(x) + ruído, com x de 0 a 10, ajuste um polinômio de grau 3 e uma spline cúbica "
     "(`bs(x, df=6)` na fórmula do statsmodels) e desenhe as duas curvas.",
     '''d = pd.DataFrame({"x": np.sort(rng.uniform(0, 10, 200))}); d["y"] = np.sin(d.x) + rng.normal(0, 0.3, 200)
poli = smf.ols("y ~ x + I(x**2) + I(x**3)", d).fit(); spl = smf.ols("y ~ bs(x, df=6)", d).fit()
plt.scatter(d.x, d.y, s=6, c="grey"); plt.plot(d.x, poli.fittedvalues, label="polinômio 3"); plt.plot(d.x, spl.fittedvalues, label="spline"); plt.legend(); plt.show()'''),

    # ---------------- 5. Classificação ----------------
    (5, "A Solução Naive", "Naive Bayes gaussiano",
     "Com os dados de câncer de mama (`load_breast_cancer`), separe 30% para teste e treine `GaussianNB`. Mostre a "
     "acurácia de teste.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(round(GaussianNB().fit(Xtr, ytr).score(Xte, yte), 3))'''),
    (5, "Discriminante Linear", "Análise discriminante linear",
     "No mesmo problema, treine `LinearDiscriminantAnalysis` e compare a acurácia de teste com o Naive Bayes.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(round(LinearDiscriminantAnalysis().fit(Xtr, ytr).score(Xte, yte), 3))'''),
    (5, "Interpretando os Coeficientes", "Logística e razão de chances",
     "Simule a inadimplência com P = logística(−3 + 0,8·dívida/renda), dívida/renda de 0 a 8. Ajuste `smf.logit` e "
     "interprete `exp(coef)` como razão de chances.",
     '''d = pd.DataFrame({"divida_renda": rng.uniform(0, 8, 2000)})
d["inad"] = (rng.random(2000) < 1 / (1 + np.exp(-(-3 + 0.8 * d.divida_renda)))).astype(int)
m = smf.logit("inad ~ divida_renda", d).fit(disp=0)
print(m.params.round(3), np.exp(m.params["divida_renda"]).round(3))  # cada +1 multiplica as chances por ~2,2'''),
    (5, "Matriz de Confusão", "Matriz de confusão e métricas",
     "Com uma regressão logística nos dados de câncer de mama (variáveis padronizadas), mostre a matriz de confusão e "
     "o `classification_report` no teste.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
print(confusion_matrix(yte, m.predict(Xte))); print(classification_report(yte, m.predict(Xte), digits=3))'''),
    (5, "Curva ROC", "Curva ROC e AUC",
     "Desenhe a curva ROC da regressão logística do exercício anterior e calcule a AUC (`roc_curve`, `roc_auc_score`).",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, p); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.show()
print(round(roc_auc_score(yte, p), 4))'''),
    (5, "Lift", "Gráfico de lift por decil",
     "Ordene os clientes de teste pela probabilidade prevista, divida em 10 decis e calcule o *lift* de cada decil "
     "(taxa de positivos no decil ÷ taxa geral).",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
y = 1 - y  # positivo = maligno
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
d = pd.DataFrame({"p": p, "y": yte}); d["decil"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
print((d.groupby("decil", observed=True).y.mean() / d.y.mean()).round(2))'''),
    (5, "Oversampling", "Classe rara: ponderação",
     "Gere um problema com 2% de positivos (`make_classification(weights=[0.98])`). Compare a revocação dos "
     "positivos de uma logística comum com uma com `class_weight=\"balanced\"`.",
     '''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
X, y = make_classification(5000, weights=[0.98], random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
for cw in (None, "balanced"):
    pr = LogisticRegression(class_weight=cw, max_iter=1000).fit(Xtr, ytr).predict(Xte)
    print(cw, "revocação", round(recall_score(yte, pr), 3), "precisão", round(precision_score(yte, pr), 3))'''),

    # ---------------- 6. Aprendizado de máquina estatístico ----------------
    (6, "Padronização", "KNN com e sem padronização",
     "Nos dados de vinhos, compare a acurácia (validação cruzada de 5 lotes) do KNN (k = 5) com as variáveis brutas e "
     "padronizadas (`StandardScaler` num `make_pipeline`).",
     '''from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
X, y = load_wine(return_X_y=True)
print(round(cross_val_score(KNeighborsClassifier(5), X, y, cv=5).mean(), 3),
      round(cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=5).mean(), 3))'''),
    (6, "Escolhendo K", "Escolhendo k por validação cruzada",
     "Use `GridSearchCV` para escolher k entre 1 e 30 no KNN padronizado dos vinhos e mostre o melhor k e a acurácia.",
     '''from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
X, y = load_wine(return_X_y=True)
g = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": range(1, 31)}, cv=5).fit(X, y)
print(g.best_params_, round(g.best_score_, 3))'''),
    (6, "Medindo Homogeneidade", "Gini e entropia de uma divisão",
     "Escreva `gini(p)` e `entropia(p)` para uma proporção p de positivos. Para um nó com 40 positivos e 60 negativos "
     "dividido em (30+, 10−) e (10+, 50−), calcule a impureza antes e a impureza ponderada depois.",
     '''def gini(p): return 2 * p * (1 - p)
def entropia(p): return 0 if p in (0, 1) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
for f in (gini, entropia):
    antes = f(40 / 100); depois = 40 / 100 * f(30 / 40) + 60 / 100 * f(10 / 60)
    print(f.__name__, round(antes, 3), round(depois, 3))'''),
    (6, "Floresta Aleatória", "Árvore × floresta",
     "Nos dados de câncer de mama, compare a acurácia (validação cruzada) de uma árvore de decisão e de uma floresta "
     "aleatória com 300 árvores.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X, y = load_breast_cancer(return_X_y=True)
for m in (DecisionTreeClassifier(random_state=0), RandomForestClassifier(300, random_state=0)):
    print(type(m).__name__, round(cross_val_score(m, X, y, cv=5).mean(), 3))'''),
    (6, "Importância da Variável", "Importância por permutação",
     "Treine uma floresta nos dados de câncer de mama e mostre as 5 variáveis mais importantes por "
     "`permutation_importance` no conjunto de teste.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
d = load_breast_cancer(as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.3, random_state=0)
rf = RandomForestClassifier(300, random_state=0).fit(Xtr, ytr)
imp = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0)
print(pd.Series(imp.importances_mean, d.data.columns).sort_values(ascending=False).head(5).round(4))'''),
    (6, "XGBoost", "Boosting com HistGradientBoosting",
     "Use `HistGradientBoostingClassifier` (o *gradient boosting* do scikit-learn, parecido com o XGBoost) nos dados de "
     "câncer de mama e compare com a floresta. Depois teste `learning_rate` de 0,01, 0,1 e 0,5.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
X, y = load_breast_cancer(return_X_y=True)
for lr in (0.01, 0.1, 0.5):
    print(lr, round(cross_val_score(HistGradientBoostingClassifier(learning_rate=lr, random_state=0), X, y, cv=5).mean(), 3))'''),
    (6, "Hiperparâmetros e Validação Cruzada", "Busca de hiperparâmetros",
     "Com `RandomizedSearchCV` (20 combinações), ajuste `max_depth`, `learning_rate` e `max_iter` do "
     "HistGradientBoosting e mostre a melhor combinação.",
     '''from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
X, y = load_breast_cancer(return_X_y=True)
busca = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0),
    {"max_depth": [2, 3, 5, None], "learning_rate": [0.01, 0.05, 0.1, 0.3], "max_iter": [100, 200, 400]},
    n_iter=20, cv=5, random_state=0).fit(X, y)
print(busca.best_params_, round(busca.best_score_, 3))'''),

    # ---------------- 7. Não supervisionado ----------------
    (7, "Calculando os Componentes", "PCA à mão × scikit-learn",
     "Nos dados de vinhos padronizados, calcule os componentes principais pelos autovetores da matriz de covariância "
     "e compare a variância explicada com `PCA().explained_variance_ratio_`.",
     '''from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(load_wine().data)
autoval = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
print((autoval / autoval.sum())[:4].round(3), PCA().fit(X).explained_variance_ratio_[:4].round(3))'''),
    (7, "Escolhendo o Número de Grupos", "K-médias e silhueta",
     "Nos vinhos padronizados, rode k-médias para k = 2 a 8 e calcule a silhueta média de cada k. Qual k a silhueta "
     "sugere?",
     '''from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X = StandardScaler().fit_transform(load_wine().data)
sil = {k: round(silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)), 3) for k in range(2, 9)}
print(sil, max(sil, key=sil.get))'''),
    (7, "O Dendrograma", "Dendrograma",
     "Faça o agrupamento hierárquico (ligação de Ward) de 30 vinhos padronizados com `scipy.cluster.hierarchy` e "
     "desenhe o dendrograma.",
     '''from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
X = StandardScaler().fit_transform(load_wine().data)[::6]
dendrogram(linkage(X, "ward")); plt.show()'''),
    (7, "Misturas de Normais", "Mistura de normais e BIC",
     "Gere dados de 3 normais 2D e ajuste `GaussianMixture` com 1 a 6 componentes. Escolha o número de componentes "
     "pelo menor BIC.",
     '''from sklearn.mixture import GaussianMixture
X = np.vstack([rng.normal(c, 0.8, (150, 2)) for c in ([0, 0], [4, 4], [0, 5])])
bic = {k: round(GaussianMixture(k, random_state=0).fit(X).bic(X), 1) for k in range(1, 7)}
print(bic, min(bic, key=bic.get))'''),
    (7, "Escalonando as Variáveis", "Variável dominante",
     "Rode k-médias (k = 3) nos vinhos **sem** padronizar e **padronizados**. Compare os grupos com as classes reais "
     "usando `adjusted_rand_score`. Qual variável domina sem padronização?",
     '''from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
d = load_wine(as_frame=True)
for X in (d.data.to_numpy(), StandardScaler().fit_transform(d.data)):
    print(round(adjusted_rand_score(d.target, KMeans(3, n_init=10, random_state=0).fit_predict(X)), 3))
print(d.data.var().sort_values(ascending=False).head(2))  # proline tem variância enorme'''),
]
