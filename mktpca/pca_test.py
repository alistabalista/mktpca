import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from sklearn import preprocessing

pca_data_org = pd.read_excel('d:/pcaexpample/pca_data.xlsx',
                         sheetname = 'shrunk',header=0,index_col=0)

pca_data_org=pca_data_org.fillna(method='backfill')
endeks = pca_data_org.index
pca_data_org['USDTRY'] = 1/pca_data_org['USDTRY']
pca_data_org['VIX'] = 1/pca_data_org['VIX']
XU100 = 1/pca_data_org['XU100']
pca_data_org=pca_data_org.drop('XU100',1)
pca_data=preprocessing.scale(pca_data_org)
sklearn_pca = sklearnPCA(n_components=1)
sklearn_transf = sklearn_pca.fit_transform(pca_data)
sklearn_transf = pd.DataFrame(sklearn_transf,index=endeks)


fig,ax=plt.subplots()
ax2=ax.twinx()
sklearn_transf.plot(ax=ax)
XU100.plot(ax=ax2,style='r')
plt.show()