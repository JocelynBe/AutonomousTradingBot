# Autonomous Trading Agent Project

This is a side project I have been working on for about a year. 
My goal is to create a trading bot able to autonomously trade any asset, although I have been working with market data from the Binance platform.

I have implemented everything from the ETL pipeline to the forward test framework and trading agent.

The model itself is implemented in PyTorch, based on a LSTM and custom reward functions.

Here are the key ideas I have so far implemented:
* Price normalization to avoid over-fitting
* Leveraged a third party package to compute 300+ technical indicators
* Loss function that reflects exactly the actions of the agent on the system (ignoring trading volume)
* Perfect oracle target to fine-tune the model with

As of today, the model does not perform well on the forward test. It has a tendency to converge towards not taking any action.
Current working hypotheses:
1. The input signal is insufficient to make the right decision
2. The exchange fees are too high given the signal available
3. The loss function is not well suited to the problem (exponential anti-diagonal fee penalty)


## Loss functions

### Portfolio value after actions

We define the fees matrix $\mathbf{F}$ as:

```math
\mathbf{F}_{i, j} = 1 \text{ if } i = j \text{ else } (1 - \eta)
```

then the conversion rate from currency i to j at time step n:
```math
\mathbf{C}^n_{i, j} = \left(\mathbf{P}^n_{i, j}\right)^{-1}
```

and finally the decision matrix $\mathbf{D}^n$ as:
```math
\mathbf{D}^n_{i, j} = \text{proportion of currency i being transferred to currency j at step n}
```

This decision matrix is the output of the ML model.

Finally, the value of the porfolio at time step n, given the initial value $p^0$ is:

```math
p^n = (p^0)^\top \cdot \prod_{k=0}^{n-1} \mathbf{F} \odot \mathbf{C}^k \odot \mathbf{D}^k
```

The model is trained to maximize $p^n$ by adjusting its decisions $\mathbf{D}^k$.

### Oracle target

Interestingly, we can find the optimal decisions for the above formula by considering the directed graph corresponding to the product of matrices.
The nodes are the values of the portfolio at each time step, and the edges are the conversion rates (including fees) between two consecutive time steps.
By taking the log of the weights, the optimal decisions are obtained by finding the shortest weighted path from the initial value to the final value. The log allows to go from the multiplication to a sum, which I think is an elegant trick.

I then tried to train the model to match the realized portfolio from the oracle decisions (instead of the transitions which are two noisy).