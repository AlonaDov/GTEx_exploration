# GTEx_exploration
Analysis of RNA-seq data from Genotype Tissue Expression project for genes, containing autism-related repeat expansions.

This project's background is a genetic study, published by Trost B., et al., in Nature, 2020 (Trost B, Engchuan W, Nguyen CM, Thiruvahindrapuram B, Dolzhenko E, Backstrom I,
et al. Genome-wide detection of tandem DNA repeats that are expanded in autism. Nature [Internet]. 2020;586(October). Available from: http://dx.doi.org/10.1038/s41586-020-2579-z). The study shows expansions of tandem DNA repeats (STRs) in 55 distinct genes which were found enriched in autistic subjects. In this project publicaly available RNA-seq data from 13 brain tissues from Genotype Tissue Expression project (GTEx, Aguet F, Brown AA, Castel SE, Davis JR, He Y, Jo B, et al. Genetic effects on gene expression across human tissues. Nature. 2017;550(7675):204–13.), combined with an output of STRs genotyping at selected loci, are utilized to answer the following questions:

Q1: Out of 55 genes with tandem DNA repeats (STRs) what are the genes, clusterizing the samples by tissues most?
Q2: What are the relationships between gene expression and features, available for the subjects?
a: Is differential gene expression associated with gender?
b: Is differential gene expression associated with age?
c: Is differential gene expression associated with death hardness? 
d: Is differential gene expression associated with ischemic time?
Q3:How would selected STR expansions affect gene expression?

Note: in this project STRs genotyping results, obtained by TCAG staff at Peter Gilgan Center for Research and Learning, Toronto, are utilized. However, due to confidentiality reasons and yet unpublished results, the identifiers of genotyped STR loci used in the project are ommited. Current project therefore serves for mere demonstrative purposes and allows to share potential predictive ability.

The following files are available in the repository:

Source code of preformed anaysis: 
GTEX_exploration.py or GTEX_exploration.ipynb (jupyter notebook)

Input files used in the analysis:
1. 'brain_gene_normCountsNonZero_selected55.tsv' -- expression data across 13 brain tissues, normalized by relative log expression approach, where genes with zero expression across samples have been removed.
2. 'GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt' -- publicaly available GTEx data with phenotypes of involved subjects: https://gtexportal.org/home/datasets
3. 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt' -- publicaly available GTEx data with attributes of involved subjects: https://gtexportal.org/home/datasets
4. 'STRs_genotypes.txt' -- STR genotypes across GTEx subjects; 15 selected STR loci are given with omitted identifiers.

5. GTEx_Analysis_v8_Annotations_SubjectPhenotypesDD.xlsx -- this file explains features, given in 'GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt' and is publically available at https://gtexportal.org/home/datasets
6. GTEx_Analysis_v8_Annotations_SampleAttributesDD.xlsx -- this file explains features, given in  'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt' and is publically available at https://gtexportal.org/home/datasets
