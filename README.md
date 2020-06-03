# Dimensionality reduction & intrinsic dimensionality estimation 

## Requirements
To use this project, the `python 3.7.0+` and following packages are required:
```
numpy 1.16.4
scipy 1.2.1
scikit-learn 0.21.1
matplotlib 3.0.3
```

## Using
1. Load data. You can use dataset from `./data` directory or load your own:
```python
import src.data as data
source_data, markers = data.load_data("data/ocr.dat", 1, False)
```

2. Perform dimensionality reduction, e.g. PCA:
```python
import src.pca as pca
pca_data, pca_error = pca.calculate(source_data)
```

3. Calculate estimation, e.g. Hubert Г statistic:
```python
import src.hubert as hubert

source_hubert = []
pca_hubert = []
for i in range(1, len(source_data)+1):
    source_hubert.append(hubert.calculate(source_data[:i], markers))
    pca_hubert.append(hubert.calculate(pca_data[:i], markers))
```

4. Draw estimation results:
```python
import src.drawer as drawer

drawer.plot_multiline(
    [source_hubert, pca_hubert], 
    label='Normalized Hubert Г statistic', 
    legend=['SRC', 'PCA'],
    ).show()
```

