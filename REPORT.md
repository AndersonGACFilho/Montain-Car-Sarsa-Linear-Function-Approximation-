# Relatório

## 1. Visão geral da grade

![Mediana de passos por configuração](results/heatmap_median_steps.png)

Mediana de passos até 10 sucessos (11 seeds por célula):

| α \ centros |      9 |     16 |     25 |     36 |        49 |
|-------------|-------:|-------:|-------:|-------:|----------:|
| 0,005       | 37.861 | 44.295 | 25.037 | 17.507 |    12.684 |
| 0,01        | 23.447 | 24.121 | 13.881 |  9.588 |     7.537 |
| 0,02        | 14.491 | 13.378 |  8.231 |  5.933 |     4.601 |
| 0,05        |  8.382 |  6.462 |  4.311 |  3.397 |     2.739 |
| 0,1         |  6.755 |  4.005 |  2.957 |  2.457 | **2.387** |

A diminuição do número de passos segue um padrão: 
    
> **Aumentar α ou o número de centros reduz o número de
    passos**. 

Da pior para a melhor configuração, a diferença é de cerca de 18×
(44.295 → 2.387). 

A única quebra clara a essa regra e **16 centros ser pior que
9 com α baixo** (0,005 e 0,01), discutida na seção 5.

## 2. Boxplot das 5 melhores configurações

As cinco configurações com menor mediana de passos (critério padrão de `plot.py`):

![Boxplot das 5 melhores configurações](results/boxplot_top5.png)

| Configuração         | Sucesso | Mediana | Média | Desvio | IQR |
|----------------------|--------:|--------:|------:|-------:|----:|
| α = 0,1, 49 centros  |   11/11 |   2.387 | 2.364 |    177 | 191 |
| α = 0,1, 36 centros  |   11/11 |   2.457 | 2.493 |    186 | 333 |
| α = 0,05, 49 centros |   11/11 |   2.739 | 2.763 |    176 | 264 |
| α = 0,1, 25 centros  |   11/11 |   2.957 | 2.998 |    174 | 239 |
| α = 0,05, 36 centros |   11/11 |   3.397 | 3.379 |     95 |  75 |

**Por que essas configurações são interessantes:**

- **Estão todas no canto da grade, em sua maioria, possuem muitos centros.** 
A melhor é exatamente o extremo (α = 0,1, 49 centros). 
Isso mostra que o ótimo provavelmente está **fora da região explorada**. 
A próxima varredura deveria testar mais de 49 centros.
- **Os ganhos estão saturando.** De 36 para 49 centros com α = 0,1, a mediana cai só
70 passos (≈3%), menos que o desvio padrão entre seeds. 
As caixas das duas primeiras configurações se sobrepõem bastante
com 11 seeds não dá para afirmar que uma é melhor que a outra. 
Já perto de 2.400 passos o agente gasta, na mediana,  14 episódios 
(≈4 falhas de 200 passos + 10 sucessos de ≈150 passos), então resta
pouco a ganhar com esta métrica.
- **A variância cresce junto com o passo.** A configuração mais conservadora do
top 5 (α = 0,05, 36 centros) tem o menor IQR (75 passos), enquanto as com α = 0,1
têm IQR de 190 a 330. Em toda a grade, o coeficiente de variação mais alto
(12%) aparece em α = 0,1 com 9 centros. É o sinal esperado de que passos de
atualização grandes deixam o aprendizado mais sensível à seed.

![Curvas de aprendizado das 5 melhores](results/learning_curves_top5.png)

As curvas de aprendizado mostram o mesmo ordenamento: as duas melhores configurações
começam a encurtar os episódios já por volta do 4º episódio, enquanto α = 0,05 com
36 centros só começa por volta do 7º.

## 3. Conclusão

Todas as 25 configurações resolveram a tarefa nas 11 seeds. A melhor foi
**α = 0,1 com 49 centros RBF** (mediana de 2.387 passos), seguida de perto por
α = 0,1 com 36 centros. O desempenho melhora de forma consistente com α e com o
número de centros, mas grande parte do efeito dos centros vem do aumento do passo
efetivo de atualização causado pelas features não normalizadas. Os próximos passos
sugeridos são: 
1. Estender a grade para α > 0,1 e mais centros, 
2. Normalizar as features para separar resolução de passo efetivo e
3. Variar σ junto com o número de centros.
