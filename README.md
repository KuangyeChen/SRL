## Core

### Database Form
A database should contain three files:<br>
data.txt: each row is a triple. e.g. `Disease_or_Syndrome Affects Plant`<br> 
entities.txt: each row is a entity name. e.g. `Plant`<br>
relations.txt: each row is a relation name. e.g. `Affects`<br>

See the data directory.
### Database Class
Class is `core.knowledge_graph.KnowledgeGraph`<br>
Usage example<br>
```python
from core.knowledge_graph import KnowledgeGraph
database = KnowledgeGraph()
database.read_data_from_txt('data/kin')
```


## Scripts
Scripts will not be made as command line apps, 
since they will eventually be merged with GUI.<br>

### Tensor Factorization
Run script `./tensor_factorization.py`<br>
Change hyperparameters directly in scripts.

### Neural Networks
* ER-MLP: run `./train_er_mlp.py`
* NTN: run `./train_ntn.py`

Change hyperparameters directly in scripts.<br>
During the training, run `tensorboard --logdir=tmp/log/` for monitoring.