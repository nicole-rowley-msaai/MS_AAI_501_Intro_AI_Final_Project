# MS_AAI_501_Intro_AI_Final_Project

**Introduction**
Financial fraud remains a major challenge for banks and e-commerce platforms as digital
transactions continue to grow. Fraud detection systems must identify rare fraudulent transactions
among millions of legitimate purchases while minimizing disruptions for normal customers. A
major difficulty is extreme class imbalance, where fraudulent transactions represent only a small
fraction of the data. Because rule-based systems struggle to adapt to evolving fraud strategies,
machine learning approaches are increasingly used to identify patterns in large transaction
datasets.

**Objectives**
This project proposes a hybrid machine learning pipeline using a dataset of
approximately 550,000 transaction records. The system will combine supervised classification
models and unsupervised anomaly detection to detect both known and emerging fraud patterns.
The supervised component will compare Random Forest and Gradient Boosting classifiers to
predict the probability that a transaction is fraudulent. These ensemble models are well suited to
structured financial data because they capture nonlinear feature relationships and perform well
with high-dimensional datasets. The system will also apply Isolation Forest or k-means
clustering to identify anomalous transactions that may represent new fraud patterns that are not
present in the labeled training data.

