## Core
**core.py** will be removed.<br>
Functionality is separated.
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

## Scripts for Tensor Factorization
Scripts will not be made as command line apps, since they will eventually be merged with GUI.<br>
**cross_validation_srl.py** will be removed.

Run script `./tensor_factorization.py`<br>
Change hyperparameters directly in script.

For neural networks use Jupyter notebook instead.


## Jupyter Notebook
Interactive environment is more suitable for machine learning and 
neural networks programming, especially during training and 
hyperparameters tuning. Neural networks are hard to train.<br>
All neural networks' training and testing codes are now in the notebook
`Neural_Networks.ipynb`

Run cells for corresponding neural networks.<br>
For babysitting during the training, run `tensorboard --logdir=tmp/log/`
