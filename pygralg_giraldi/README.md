# pyGralg

## Performances (Class stratified genetic optimization. Optimized Informedness)
| Dataset (Subsample)        | Test Sec Acc | Alphabet     | Selected Alphabet | First GA Time | Second GA Time | Test Time  | Workstation  |
|----------------------------|--------------|--------------|-------------------|---------------|----------------|------------|--------------|
| AIDS (10%)                 | 0.99 ± 0.0   |53.33 ± 57.33 |  2.0 ± 1.41       | 32.51 ± 6.36  | 0.01 ± 0.0     |1.05 ± 0.02 |    i9        |
| AIDS (30%) 		         | 0.99 ± 0.0   |47.33 ± 22.81 |  1.33 ± 0.47      | 61.64 ± 21.35 | 0.01 ± 0.0     |1.05 ± 0.02 |    i9        |
| AIDS (50%)                 | 0.99 ± 0.0   |79.33 ± 62.19 |  1.67 ± 0.94      | 127.26 ± 16.16| 0.01 ± 0.01    |1.07 ± 0.01 |    i9        |
| GREC (10%) 		         | 0.82 ± 0.01  |271.67 ± 5.73 |  86.67 ± 4.11     | 35.58 ± 0.56  | 0.62 ± 0.02    |0.63 ± 0.0  |    i9        |
| GREC (30%)                 | 0.83 ± 0.01  |333.0 ± 6.98  |  96.67 ± 5.31     | 60.63 ± 0.33  | 0.64 ± 0.05    |0.64 ± 0.0  |    i9        |
| GREC (50%)  		         | 0.84 ± 0.0   |348.0 ± 10.61 |  127.0 ± 16.39    | 84.14 ± 1.54  | 0.87 ± 0.12    |0.65 ± 0.01 |    i9        |
| Letter1 (10%)              | 0.97 ± 0.01  | 106.0 ± 0.82 |  24.33 ± 1.7      | 17.27 ± 0.59  | 0.5 ± 0.04     |0.17 ± 0.0  |    i9        |
| Letter1 (30%)              | 0.97 ± 0.0   |144.33 ± 13.89|  31.67 ± 3.3      | 28.22 ± 0.3   | 0.7 ± 0.05     | 0.2 ± 0.03 |    i9        |
| Letter1 (50%)		         | 0.97 ± 0.0   |170.67 ± 20.98|  43.0 ± 11.58     | 37.11 ± 1.22  | 0.83 ± 0.17    |0.2 ± 0.03  |    i9        |
| Letter2 (10%)              | 0.9 ± 0.01   |128.33 ± 17.78|  44.67 ± 8.26     | 18.84 ± 0.65  | 1.16 ± 0.35    | 0.18 ± 0.0 |    i9        |
| Letter2 (30%)              | 0.92 ± 0.0   |209.0 ± 3.56  |  74.33 ± 5.73     | 31.23 ± 0.4   | 1.92 ± 0.13    | 0.19 ± 0.0 |    i9        |
| Letter2 (50%)              | 0.92 ± 0.0   |268.0 ± 15.9  |  81.67 ± 1.7      | 42.74 ± 0.33  | 2.14 ± 0.21    | 0.19 ± 0.0 |    i9        |
| Letter3 (10%)              | 0.89 ± 0.0   | 201.0 ± 4.97 |  82.67 ± 2.87     | 25.68 ± 0.57  | 2.47 ± 0.11    | 0.2 ± 0.0  |    i9        |
| Letter3 (30%) 	         | 0.88 ± 0.02  |293.0 ± 10.2  |  126.67 ± 10.96   | 44.9 ± 0.54   | 3.71 ± 0.3     | 0.21 ± 0.0 |    i9	    |
| Letter3 (50%)		         | 0.9 ± 0.0    |318.67 ± 15.15|  133.0 ± 7.26     | 62.83 ± 1.65  | 4.08 ± 0.16    | 0.22 ± 0.0 |    i9        |
| Mutagenicity (10%)         | 0.69 ± 0.0   |356.33 ± 64.88|  147.67 ± 49.87   | 425.61 ± 23.02| 0.31 ± 0.09    |3.85 ± 0.03 |    i9        |
| Mutagenicity (30%) 	     | 0.7 ± 0.01   |309.0 ± 34.07 |  98.33 ± 14.06    | 963.45 ± 49.59| 0.23 ± 0.02    |3.88 ± 0.02 |    i9	    |
| Mutagenicity (50%)		 | 0.69 ± 0.01  |389.67 ± 43.25|  126.33 ± 21.48   |1579.76 ± 21.89| 0.31 ± 0.06    |3.94 ± 0.07 |    i9        |
| AIDS (clique)              |  0.99 ± 0.01 |91.0 ± 97.58  |  5.67 ± 5.25      | 109.04 ± 8.72 |  0.04 ± 0.02   |1.71 ± 0.1  |    i9        |
| GREC (clique)              |  0.91 ± 0.0  |160.0 ± 12.83 |  45.0 ± 6.98      | 82.09 ± 3.64  |  0.36 ± 0.03   |1.68 ± 0.0  |    i9        |
| Letter1 (clique)           | 0.96 ± 0.0   |74.0 ± 3.56   |  19.33 ± 2.05     | 37.89 ± 0.34  | 0.55 ± 0.03    |0.36 ± 0.01 |    i9        |
| Letter2 (clique)           | 0.73 ± 0.02  |70.33 ± 2.36  | 32.67 ± 1.89      | 36.45 ± 0.74  |1.14 ± 0.01     | 0.36 ± 0.01|    i9        |
| Letter3 (clique)           | 0.73 ± 0.03  |94.33 ± 10.53 | 53.67 ± 7.41      | 43.13 ± 2.98  |1.44 ± 0.24     | 0.38 ± 0.01|    i9        |
| Mutagenicity (clique)      | 0.73 ± 0.01  |144.33 ± 4.19 | 45.67 ± 9.18      | 360.13 ± 12.18|0.15 ± 0.03     | 5.57 ± 0.19|    i9        |

Add a local metric learning method based on evolutionary strategy.

Dependencies (all of which can be downloaded from `pip`/`conda`, except where noted):
* `joblib` v. 0.12
* `Graph_Sampling` must be downloaded from https://github.com/Alessi0X/Graph_Sampling
* `scikit-learn`
* `deap`
* `numpy`
* `scipy`
* `networkx`

ToDo:
* spostare le strategie di random sampling in extractorStrategies
* valutare la parallelizzazione delle strategie di random sampling