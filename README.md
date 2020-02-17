# PAKDD-20-JarKA
Pytorch implemetation of PAKDD'20---"JarKA: Modeling Attribute Interactions for Cross-lingual Knowledge Alignment"

## Whole Framework
<div align=center><img width="600" height="600" src="./model.png"/></div>

## Dependencies

* Python 3.6
* Pytorch "1.1.0"
* CPU/GPU


## How to run

```bash
Comming soon...
```

## Extensive Experiments

### Dataset discription

We evaluate the proposed model on three datasets. One is a well-adopted public dataset named DBP15K, the other two, named DBP15K-1 and DBP15K-2, modify DBP15K. DBP15K-1 loosens the constraint as 2 self-contained relationship triplets, and DBP15K-2 further removes the self-contained constraint. Thus, the clustering of the three datasets is different. Table 1 shows the statistics of the datasets.


<div align=center><img height="200" src="./data.png"/></div>

### Overall experiment results

<div align=center><img src="./overall_res.png"/></div>

### Sensitivity to Graph Clustering
We compare JarKA and BootEA on three ZH-EN datasets, which shows that both of them perform poorer when the clustering coefficient (cc) of the dataset is smaller. But their performance gap increases with the decrease of cc, indicating BootEA is more sensitive to the clustering characteristics of the graph than JarKA, as BootEA only models the structures.

<div align=center><img width="50%" height="50%" src="./cc.png"/></div>


