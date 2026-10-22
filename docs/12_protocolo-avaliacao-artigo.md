# Protocolo de avaliação do artigo (Ferrari, Leandro & Coelho, 2025)

Como o artigo mede o ARMAX/MOGWO, lido direto do código MATLAB dos
autores em `data/raw/WRSO and NSWRSO_draft_all/` (pasta
`Multi_objective_for_MIMO_systems/MOGWO/`). O objetivo é saber **o que
os números das Tabelas 2 e 3 significam** antes de comparar com os
nossos modelos. Ficha do artigo: `0_referencias-tcc.md` §1. Recorte dos
dados e ARMAX no dataset completo: `2_split_artigo_vs_dataset_completo.md`.

## 1. Resposta curta

O artigo faz **previsão um passo à frente (h = 1 h), sempre alimentada
com os dados medidos**. Na identificação de sistemas, isso se chama
modo série-paralelo, ou *predictor*. **Não há free-running**: o modelo
nunca reaproveita as próprias previsões como entrada, nem na estimação
nem na validação.

| Item | Valor no artigo |
|---|---|
| Janela (lags) | 3 amostras (3 h) de `y`, `u` e `e` |
| Horizonte | 1 passo (1 h) |
| `y` nos lags | **medido** (não previsto) |
| `u` nos lags | medido, só `u[k-1..k-3]`: o modelo não usa `u[k]` |
| Inicialização | `ŷ[1..3] = y[1..3]` (copiado do dado real, erro zero) |
| Estimação | linhas MATLAB `550:1150` (601 amostras) |
| Validação | linhas MATLAB `1500:2000` (501 amostras), mesma função one-step |
| Métricas | MSE e R² **por saída**, além dos agregados `MSE₃`/`R²₃` |

## 2. Evidência no código

Em `fun_objetivo.m` (usada na otimização) e `fun_objetivo_gera_res.m`
(métricas de estimação e validação), todos os regressores são versões
atrasadas da série **medida**:

```matlab
y1_1 = [0 Y(1,1:tam-1)]';     % y1[k-1] medido
u1_1 = [0 U(1,1:tam-1)]';     % u1[k-1] medido
Yw1 = -y1_1*a11 - y1_2*a21 - y1_3*a31 + Y2 + Y3 + U1 + U2 + U3 + U4;
```

A previsão `Yw` é calculada para a série inteira em uma única operação
vetorizada. Não existe laço que realimente `Yw[k-1]` no cálculo de
`Yw[k]`.

### O termo MA (coeficientes `c`)

```matlab
E1  = Y(1,:)' - Yw1;                          % resíduo da parte ARX
Ew1 = e1_1*ey11 + e1_2*ey12 + e1_3*ey13;      % lags de E1
MSE1 = sum((E1 - Ew1).^2) / tam;              % = mean((y - (Yw + Ew)).^2)
```

- A previsão final é `ŷ = Yw + Ew` (`fun_objetivo_gera_curva.m`).
- O erro `e[k-j]` que entra no MA é o **resíduo da parte ARX**
  (`y − Yw`), calculado de uma vez a partir do dado medido. Num ARMAX
  de livro, ele seria o resíduo do modelo completo (`y − Yw − Ew`),
  calculado de forma recursiva. É uma simplificação, mas o resultado
  continua sendo previsão um passo à frente.
- O MOGWO avalia uma partícula por vez (`fun_objetivo(..., 1)` em
  `MOGWO.m`), então a indexação linear `E1(1:tam-1)` em
  `fun_objetivo.m` não mistura soluções diferentes.

### Escolha do "melhor modelo"

Em `gera_result.m` (linhas 41–43), a solução escolhida na frente de
Pareto é a que fica mais próxima de R² = 1 nas três saídas **dos dados
de validação**:

```matlab
dis = sqrt((sum(ones(3,10000) - R2_val).^2));
xbest = PS(:,pos);
```

Por isso, as linhas "The best model from validation" da Tabela 2 e os
coeficientes da Tabela 3 foram **selecionados no conjunto em que são
avaliados**. Não são uma avaliação fora da amostra limpa. As linhas
μ/σ da Tabela 2 (média sobre toda a frente de Pareto das 100 rodadas)
não têm esse viés de seleção.

## 3. O que isso significa para comparar

1. **R² ≈ 0,94 em vazão e nível é um número fácil de obter em h = 1.**
   Com `y[k-1]` medido como entrada, um modelo só precisa prever a
   variação de uma hora. O teto real do ARMAX aparece no free-run:
   `2_split_artigo_vs_dataset_completo.md` §7 mostra o ARMAX/MOGWO caindo
   de 0,915/0,932/0,671 (1 passo) para 0,570/0,293/0,148 (12 passos) no
   teste 70/15/15.
2. **Os nossos modelos fazem forecast de 12 h a partir de uma origem**
   (`11_modelo-final.md` §2: janela de 168 h, entradas futuras
   conhecidas). Comparar o R² médio de 12 h do nosso modelo com a
   Tabela 2 mistura horizontes e **subestima** o nosso modelo.
3. **Comparação justa com o artigo: use o passo 1 do nosso horizonte.**
   Em `11_modelo-final.md` §5.3, o R² médio das 3 saídas no passo 1 é:

   | Modelo | R² médio, passo 1 (teste) |
   |---|---|
   | ARMAX/MOGWO (Tabela 3), mesmo conjunto | 0,83 |
   | SpectralTFNet final, `full` | 0,892 |
   | SpectralTFNet final, `pumps` | 0,883 |

   No passo 1 o nosso modelo recebe `u[k]` (entrada futura conhecida),
   que o ARMAX não usa. No `pumps` a vantagem se limita às bombas.
4. **Para comparar com os números publicados**, avalie o nosso modelo
   em h = 1 nas linhas `1499:2000` (slice Python) e reporte R² por
   saída. Cada origem do nosso modelo já parte do `y` medido até `k-1`,
   então o passo 1 está nas mesmas condições do artigo.
5. **Para comparar horizontes longos**, use o ARMAX em free-run (o
   modo "12 passos" de `2_split_artigo_vs_dataset_completo.md` §7). Esse
   resultado é nosso, não do artigo.
6. **Baseline de persistência** (`ŷ[k] = y[k-1]`): em h = 1 ele é a
   referência mínima e deve aparecer ao lado do ARMAX em qualquer
   tabela de 1 passo. Ainda não foi calculado nos intervalos do artigo.

## 4. Resumo para citar no texto

> O ARMAX/MOGWO de Ferrari et al. (2025) é avaliado em predição um
> passo à frente (1 h) com saídas medidas nos regressores, inclusive na
> validação, e o modelo reportado foi selecionado pelo R² de
> validação. Os R² publicados (0,948 / 0,953 / 0,836 no melhor modelo)
> são, portanto, de previsão a 1 h e não indicam a qualidade em
> simulação livre ou em horizontes maiores. No nosso conjunto de
> teste (split cronológico 70/15/15), os mesmos coeficientes caem de
> 0,915 / 0,932 / 0,671 em 1 passo para 0,570 / 0,293 / 0,148 em 12
> passos free-run.
