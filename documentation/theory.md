# Definitions

## Portfolio

The portfolio $p$ is modeled as a vector: [amount of currency 0, amount of currency 1, etc]
We can further track that over time, such that if $n$ denotes the current time step:
```math
p^n_i = \text{amount of currency i owned at step n}
```
## Decisions

A decision at time step n is modeled as a square matrix $\mathbf{D}^n$ of size N_CURRENCIES with values in $\[0, 1\]$
such that:

```math
\mathbf{D}^n_{i, j} = \text{proportion of currency i being transferred to currency j at step n}
```

With the constraint that:

```math
\forall i, \sum_k \mathbf{D}^n_{i, k} = 1
```

Provided that all prices are equal to 1 and there are no fees we have:

```math
\begin{align}
p^{n+1}_i &= \sum_k p^n_k \mathbf{D}^n_{k, i} \\
          &= \langle p^n \; , \; \mathbf{D}^n_{\cdot, i} \rangle
\end{align}

```
    

NB: We hypothesize here that we have perfect liquidity which is a reasonable approximation to make
    given the large volumes and the small size of the portfolio ~ 1k at this point in time 

and by extension:

```math
p^{n+1} = (\mathbf{D}^n)^\top \cdot p^n
```
    
    
## Fees

Still assuming all prices are equal to but that there is a fee proportional to each transaction by a factor $\eta$: 

```math
p^{n+1}_i = \sum_k p^n_k \mathbf{D}^n_{k, i} (1 - \eta + \delta_{i, k} \eta)
```

where $\delta_{i, k} = 1 \text{ if } i = k \text{ else } 0$

We further define the fees matrix $\mathbf{F}$ as:

```math
\mathbf{F}_{i, j} = 1 \text{ if } i = j \text{ else } (1 - \eta)
```
    
    
NB: The fees are technically not time dependant, indeed some exchanges will lower the fees after a large enough volume of transactions. 
However for our purposes at this time, this is an acceptable simplification. 
    
We get: 

```math
p^{n+1}_i = \sum_k p^{n}_k \mathbf{D}^n_{k, i} \mathbf{F}_{k, i}
```
    

and with the Hadamard product $\odot$ we get:
```math
p^{n+1} = F^\top \odot (\mathbf{D}^n)^\top \cdot p^n
```
    
    
## Price and conversion rate

Finally, we introduce the notion of conversion rate.
At every time step we assume that there exist a price matrix $\mathbf{P}^n_{i, j}$ such that at every time step n,
assuming that there are no fees, we can convert a quantity $q_j$ of currency j to a quantity $q_i$ of same value:

NB: we make a strong hypothesis here that such a price exists when the exchange 
    is actually based on a bidding system relying on an orderbook. 

```math
\begin{align}
q_j &= q_i \left(\mathbf{P}^n_{i, j}\right)^{-1} \\
q_i &= q_j \mathbf{P}^n_{i, j} \\
\mathbf{P}^n_{j, i} &= \left(\mathbf{P}^n_{i, j}\right)^{-1}
\end{align}
```

For example, say the price of Bitcoin in USD is 10k then: q_BTC = q_USD / 10k

Thus, we define the conversion rate matrix $\mathbf{C}$ as:

```math
\mathbf{C}^n_{i, j} = \left(\mathbf{P}^n_{i, j}\right)^{-1}
```

such that:

```math
q_i * \mathbf{C}^n_{i, j} = q_j
```

Putting everything together:

```math
\begin{align}
p^{n+1}_i &= \sum q_k * \mathbf{C}^n_{k, i} \\
          &= \sum_k p^n_k * \mathbf{D}^n_{k, i} * \mathbf{F}_{k, i} * \mathbf{C}^n_{k, i}
\end{align}
```

Which gives us this nice formula:
```math
p^{n+1} = \mathbf{F}^\top \odot (\mathbf{C}^n)^\top ⊙ (\mathbf{D}^n))^\top \cdot p^n
```
Thanks to the commutativity of the Hadamard product, this can be wrote:

```math
p^{n+1} = \left( \mathbf{F} \odot \mathbf{C}^n \odot \mathbf{D}^n \right)^\top \cdot p^n
```

    

## Recursion

To simplify notations we write: $\mathbf{M}^n = \mathbf{F} \odot \mathbf{C}^n \odot \mathbf{D}^n$
It follows from the previous formula:
```math
\begin{align}
p^{n+1} &= (\mathbf{M}^n)^\top \cdot (\mathbf{M}^{n-1})^\top \cdot p^{n-1} \\
        &= (\mathbf{M}^{n-1} \cdot \mathbf{M}^{n})^\top \cdot p^{n-1} \\
        &= \left(\prod_{k=0}^n \mathbf{M}^k\right)^T \cdot p^0 \\
        &= (p^0)^\top \cdot \left( \prod_{k=0}^n \mathbf{M}^k \right) \\
\end{align}
```
    

Finally:
```math
p^n = (p^0)^\top \cdot \prod_{k=0}^{n-1} \mathbf{F} \odot \mathbf{C}^k \odot \mathbf{D}^k
```
    

# Model

Given a training sequence of length N, we train the model to make decisions that maximize the value of $p^N$.

## Value function

Assuming currency 0 is the currency of reference, we define the value $\phi$ of a vector $p^n$ as:

```math
\begin{align}
\phi(p^n, \mathbf{P}_{0, \cdot}) &= \sum_k p_k^n * \mathbf{P}_{0, k} \\
                                 &= \langle p^n \; , \; \mathbf{P}_{0, \cdot} \rangle \\
\end{align}
```
    

NB: we purposefully omit the fee and assume perfect liquidity.

## Optimization

Given the previous definitions we can write:

```math
\begin{align}
\phi(p^N, \mathbf{P}_{0, \cdot}) &= \langle p^N , \left( \mathbf{P}^N_{0, \cdot} \right) \rangle \\
                                 &= \langle 
                                        (p^0)^\top \cdot \left( \prod_{k=0}^{N-1} \mathbf{F} \odot \mathbf{C}^k \odot \mathbf{D}^k \right)
                                        \; , \; 
                                        \mathbf{P}^N_{0, \cdot} 
                                    \rangle
\end{align}
```

In the above only 
```math
\left( \mathbf{D}^n \right)_{n=0}^{N-1}
```
is learnable, which we train to optimize $\phi(p^N, \mathbf{P}_{0, \cdot})$

## Features

There are currently two types of features as input of the model:
* Normalized candles
* Technical indicators computed with the `jesse` package

## Price to simulate transactions

One important thing to be mindful of is the information available by the model at inference time. 
We can make the decision at two different moments, either at the beginning of a time step or at the end of one.
At the beginning of a time step we know the full candles for previous time steps and only the open price of the current 
time step. At the end of a time step, we know the full candles for previous time steps and the current time step. 

In both scenarios, the question is which price to use for the transaction:
* Decision at beginning of time step: we can use the open price of the time step
* Decision at the end of time step: we can use the close price of the time step or the open price of the next time step.

For simplicity, I am going to assume we make the decision at the end of a time step. 
However, in order to make the model robust to price changes, I am going to use the open price of the next time step. 

Finally, when we compute the final gain we will use the last known close price. Ideally we would want to use the next 
open price as well but that would require adding one more time step than the features which is annoying technically. 

Based on an analysis of `Binance_BTCUSDT_minute.csv`, the close price of a time step is 97.5% between the low and high 
of the next time step. This justifies using the close price as the price to buy and sell at.

NB: Using the close price is probably suboptimal. Ideally, we would want the price to be closest to the next low 
when buying and closest to the next high when selling. This can be meaningful oftentimes the relative difference 
between low and high can be greater than the fee of 0.1%.
