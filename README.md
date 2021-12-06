# LDC-MGM
Molecular Clump extraction algorithm based on Local Density Clustering

*Note* The core idea of the algorithm comes from [this paper](https://ui.adsabs.harvard.edu/abs/2014Sci...344.1492R/abstract)
```
Rodriguez A ,  Laio A . Clustering by fast search and find of density peaks[J]. Science, 2014, 344(6191):1492.
```

## Dependencies
The code is completed with Python  3. The following dependencies are needed to run the code:

* numpy~=1.19.2
* pandas~=1.3.2
* tabulate~=0.8.9
* matplotlib~=3.3.4
* scikit-image~=0.18.1
* scipy~=1.6.2
* astropy~=4.2



# Install
I suggest you install the code using pip from an Anaconda Python 3 environment. From that environment:
```
git clone https://github.com/Luoxiaoyu828/LDC-MGM.git
cd LDC-MGM/dist
pip install DensityClust-1.0.7.tar.gz
```
or you can install LDC package directly in pypi.com. using:
```
pip install DensityClust==1.0.7
```

# Usage
```
from DensityClust import localdensitycluster_mian as LDC

data_path = r'test/data_3d.fits'
LDC.localDenCluster(data_path)

```

# Citation
If you use this code in a scientific publication, I would appreciate citation/reference to this repository. 

```
@misc{luo2021molecular,
      title={Molecular Clump Extraction Algorithm Based on Local Density Clustering}, 
      author={Xiaoyu Luo and Sheng Zheng and Yao Huang and Shuguang Zeng and Xiangyun Zeng and Zhibo Jiang and Zhiwei Chen},
      year={2021},
      eprint={2110.11620},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
      }
```
