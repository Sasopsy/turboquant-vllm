# [cite_start]TurboQuant Algorithm Specification [cite: 2, 10, 11]

## I. Mathematical Definitions & Offline Pre-computation

### 1. The Induced Coordinate Distribution
[cite_start]For any unit vector $x \in \mathbb{S}^{d-1}$ (where $||x||_2 = 1$ in $d$-dimensional Euclidean space), applying a uniformly random orthogonal rotation induces a specific probability density function on its individual coordinates[cite: 138, 143]. This is a scaled Beta distribution defined as:
[cite_start]$$f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} (1-x^2)^{(d-3)/2} \quad \text{for } x \in [-1, 1]$$ [cite: 144]

### 2. 1-Dimensional Lloyd-Max Codebook Generation
[cite_start]To quantize scalars drawn from $f_X(x)$ to a specific bit-width $b$, a codebook $C$ of $2^b$ floating-point centroids $\{c_1, c_2, \dots, c_{2^b}\}$ must be pre-computed offline[cite: 213, 214]. The centroids are found by solving the following continuous k-means optimization problem to minimize the Mean Squared Error (MSE) cost:
[cite_start]$$\mathcal{C}(f_X, b) := \min_{-1 \le c_1 \le c_2 \le \dots \le c_{2^b} \le 1} \sum_{i=1}^{2^b} \int_{\frac{c_{i-1}+c_i}{2}}^{\frac{c_i+c_{i+1}}{2}} |x-c_i|^2 \cdot f_X(x) dx$$ [cite: 221]

---

## II. Algorithm 1: $\text{TurboQuant}_{mse}$

[cite_start]Optimizes for Mean-Squared Error (MSE) reconstruction[cite: 218].

### Global Initialization
1. [cite_start]**Inputs:** Dimension $d$ and target bit-width $b$[cite: 218].
2. **Rotation Matrix:** Generate a random orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$. This is constructed by applying QR decomposition to a random matrix with i.i.d. [cite_start]Normal entries ($\mathcal{N}(0,1)$), extracting the orthogonal matrix $Q$, and setting $\Pi = Q$[cite: 209].
3. [cite_start]**Codebook:** Construct codebook $C_{mse} = \{c_1, \dots, c_{2^b}\}$ by solving the optimization problem $\mathcal{C}(f_X, b)$ defined above[cite: 218, 221].

### Procedure $\text{Quant}_{mse}(x)$
[cite_start]**Input:** $x \in \mathbb{S}^{d-1}$ [cite: 207]
1. [cite_start]$y \leftarrow \Pi \cdot x$ [cite: 218]
2. **For every** $j \in \{1, \dots, d\}$ **do:**
   [cite_start]$$idx_j \leftarrow \arg\min_{k \in [2^b]} |y_j - c_k|$$ [cite: 218]
[cite_start]**Output:** $idx \in [2^b]^d$ (An array of $d$ integers, each represented by $b$ bits) [cite: 218]

### Procedure $\text{DeQuant}_{mse}(idx)$
[cite_start]**Input:** $idx \in [2^b]^d$ [cite: 218]
1. **For every** $j \in \{1, \dots, d\}$ **do:**
   [cite_start]$$\tilde{y}_j \leftarrow c_{idx_j}$$ [cite: 218]
2. [cite_start]$\tilde{x} \leftarrow \Pi^\top \cdot \tilde{y}$ [cite: 218]
[cite_start]**Output:** $\tilde{x} \in \mathbb{R}^d$ [cite: 218]

---

## III. Algorithm 2: $\text{TurboQuant}_{prod}$

[cite_start]Provides unbiased estimations for inner products by composing a $(b-1)$-bit MSE quantizer with a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform[cite: 261, 270, 272].

### Global Initialization
1. [cite_start]**Inputs:** Dimension $d$ and target total bit-width $b$[cite: 261].
2. [cite_start]**Base Quantizer:** Instantiate $\text{TurboQuant}_{mse}$ (Algorithm 1) using bit-width set to $b-1$[cite: 261].
3. [cite_start]**QJL Projection Matrix:** Generate a random matrix $S \in \mathbb{R}^{d \times d}$ with i.i.d. entries $S_{i,j} \sim \mathcal{N}(0,1)$[cite: 261].

### Procedure $\text{Quant}_{prod}(x)$
[cite_start]**Input:** $x \in \mathbb{S}^{d-1}$ [cite: 274]
1. [cite_start]$idx \leftarrow \text{Quant}_{mse}(x)$ *(Executes Algorithm 1 using $b-1$ bits)* [cite: 261]
2. [cite_start]$r \leftarrow x - \text{DeQuant}_{mse}(idx)$ *(Computes the residual error vector)* [cite: 261]
3. [cite_start]$qjl \leftarrow \text{sign}(S \cdot r)$ *(Applies element-wise sign function for 1-bit QJL)* [cite: 261]
4. [cite_start]$\gamma \leftarrow ||r||_2$ *(Calculates the $L_2$ norm of the residual)* [cite: 261]
[cite_start]**Output:** The tuple $(idx, qjl, \gamma)$, mapping $x \rightarrow [2^{b-1}]^d \times \{-1, +1\}^d \times \mathbb{R}$[cite: 261, 274].

### Procedure $\text{DeQuant}_{prod}(idx, qjl, \gamma)$
[cite_start]**Input:** $(idx, qjl, \gamma)$ [cite: 261]
1. [cite_start]$\tilde{x}_{mse} \leftarrow \text{DeQuant}_{mse}(idx)$ *(Reconstructs the $(b-1)$-bit base vector)* [cite: 261]
2. [cite_start]$\tilde{x}_{qjl} \leftarrow \frac{\sqrt{\pi/2}}{d} \cdot \gamma \cdot S^\top \cdot qjl$ *(Reconstructs the scaled QJL residual approximation)* [cite: 261]
3. [cite_start]$\tilde{x} \leftarrow \tilde{x}_{mse} + \tilde{x}_{qjl}$ *(Aggregates approximations)* [cite: 261]
[cite_start]**Output:** $\tilde{x} \in \mathbb{R}^d$ [cite: 261]