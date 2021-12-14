import os
import sys
import seaborn as sns
import pandas as pd
import csv
import matplotlib.pyplot as plt


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.robust.scale import mad

# change working directory
os.chdir('../data/raw')

# load data
copy_loss_df = pd.read_table('copy_number_loss_status.tsv.gz', index_col=0, engine='c')   # copy loss
mutation_df = pd.read_table('pancan_mutation_freeze.tsv.gz', index_col=0, engine='c')     # mutation 0/1
rnaseq_full_df = pd.read_table('pancan_rnaseq_freeze.tsv.gz', index_col=0, engine='c')    # RNA-seq
sample_freeze = pd.read_table('sample_freeze.tsv', index_col=0, engine='c')               # patients
mut_burden = pd.read_table('mutation_burden_freeze.tsv', engine='c')                      # mutation float
cancer_genes = pd.read_table('vogelstein_cancergenes.tsv')                                # gene data     


#integrate copy number and genes
def integrate_copy_number(y, cancer_genes_df, genes, loss_df):
    genes_sub = cancer_genes_df[cancer_genes_df['Gene Symbol'].isin(genes)]
    tumor_suppressor = genes_sub[genes_sub['Classification*'] == 'TSG']
    copy_loss_sub = loss_df[tumor_suppressor['Gene Symbol']]
    copy_loss_sub.columns = [col + '_loss' for col in copy_loss_sub.columns]
    y = y.join(copy_loss_sub)
    y = y.fillna(0)
    y = y.astype(int)
    return y

x = ['TP53']
# TP53 mutation
y = mutation_df[x]
y = integrate_copy_number(y=y, cancer_genes_df=cancer_genes, genes=x, loss_df=copy_loss_df)
y = y.assign(total_status=y.max(axis=1))
y = y.reset_index().merge(sample_freeze, how='left').set_index('SAMPLE_BARCODE')
sum_df = y.groupby('DISEASE').sum()
dive = sum_df.divide(y['DISEASE'].value_counts(sort=False).sort_index(),axis=0)
dise_sele = (sum_df['total_status']> 15) & (dive['total_status'] > 0.05)
diseases = dise_sele.index[dise_sele].tolist()
y_df = y[y.DISEASE.isin(diseases)].total_status                                                # final inactivation
y_df = y_df.loc[list(set(y_df.index) & set(rnaseq_full_df.index))]
# filter rna
rna = rnaseq_full_df.loc[y_df.index, :]
# delete hypermutaion
new_mut_burden = mut_burden[mut_burden['log10_mut'] < 5 * mut_burden['log10_mut'].std()]
y_temp = new_mut_burden.merge(pd.DataFrame(y_df), right_index=True, left_on='SAMPLE_BARCODE').set_index('SAMPLE_BARCODE')

y_sub = y.loc[y_temp.index]['DISEASE']  # sample - cancer
covar_dummy = pd.get_dummies(sample_freeze['DISEASE']).astype(int)

covar_dummy.index = sample_freeze['SAMPLE_BARCODE']

covar = covar_dummy.merge(y_temp, right_index=True, left_index=True)
covar = covar.drop('total_status', axis=1)
y_df = y_df.loc[y_sub.index]
strat = y_sub.str.cat(y_df.astype(str))   # gene status 0-1

# genes
x_df = rna.loc[y_df.index, :]
#MAD
med_dev = pd.DataFrame(mad(x_df), index=x_df.columns)
mad_genes = med_dev.sort_values(by=0, ascending=False).iloc[0:keep_num].index.tolist()
x_df = x_df.loc[:, mad_genes]
# standard
fitted_scaler = StandardScaler().fit(x_df)
x_df_update = pd.DataFrame(fitted_scaler.transform(x_df), columns=x_df.columns)
x_df_update.index = x_df.index
x_df = x_df_update.merge(covar, left_index=True, right_index=True)


# train test
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=0, stratify=strat)
#save data
#x_train.to_csv("xtrain.csv")
#x_test.to_csv("xtest.csv")
y_train.to_csv("ytrain.csv")
y_test.to_csv("ytest.csv")