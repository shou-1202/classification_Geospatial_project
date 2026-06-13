# 🔥 Forest Fire Anomaly Detection: Geospatial Machine Learning

**Satellite monitoring of the Wayanad forest corridor utilizing Sentinel-2 optical data and FIRMS thermal anomaly detection.**

## 📋 Project Overview
This project builds an end-to-end geospatial machine learning pipeline to detect forest fires. The primary engineering challenge of this project is **extreme class imbalance**. In any given satellite capture, 99.9% of the forest is healthy, and 0.1% is actively burning. 

Traditional machine learning models fail in this environment—achieving 99.9% "accuracy" simply by blindly guessing "No Fire" every time. This project implements advanced resampling techniques and custom evaluation metrics to force the algorithms to prioritize the detection of rare, catastrophic anomalies.

## 🏗️ Data Architecture & Methodology
The dataset consists of multi-spectral satellite imagery (NDVI, NDWI, SWIR, etc.) engineered into an 11.5-million-row tabular format suitable for Scikit-Learn classifiers.

### Temporal Engineering: Squashing Months, Stacking Years
To handle the massive scale of pixel-by-pixel satellite data, the pipeline uses a specific temporal aggregation strategy:
* **Squashing Months:** Monthly satellite passes were aggregated (squashed) into composite metrics (e.g., maximum temperature, average NDVI). This significantly reduces atmospheric noise, such as cloud cover, and condenses the feature space.
* **Stacking Years:** Rather than flattening the entire timeline, individual years were stacked vertically. This prevents data leakage across seasons and allows the models to study the historical context of distinct dry and wet cycles over time.

## ⚖️ Handling Imbalance & The Custom F5-Metric
Because a false negative (missing a fire) results in environmental destruction, while a false positive (a false alarm) merely results in a wasted drone inspection, standard accuracy and standard F1-scores are dangerous metrics for this use case.

I engineered a custom **F5-Score** (`beta=5`) as the ultimate grading rubric for hyperparameter tuning.
* **Why Beta = 5?** This explicitly hardcodes the business logic into the algorithm's training phase: *Catching a real fire (Recall) is exactly five times mathematically more important than avoiding a false alarm (Precision).*

### Resampling Strategy
To prevent the model from ignoring the minority class during training, the pipeline implements **Stratified Splitting** followed by undersampling (and synthetic minority over-sampling/SMOTE), ensuring the algorithms train on a perfectly balanced 50/50 dataset while being evaluated on a heavily imbalanced, real-world test set.

## 🏆 Model Arena & Results
Four distinct classification architectures were pitted against each other: Logistic Regression, Support Vector Machines (LinearSVC), Random Forest, and Gradient Boosting.

As expected, linear models failed to capture the complex, non-linear relationships between moisture/vegetation indices and fire events. 

**The Champion: Tuned Gradient Boosting Classifier**
* **Recall:** **72.0%** (Successfully identified ~72% of all actual fires in a purely pixel-by-pixel analysis).
* **Precision:** 2.4% (Expected drop due to extreme base-rate imbalance; optimized purely for anomaly flagging).
* **F5-Score:** 0.3428

## 🚀 Future Scope
While this tabular, pixel-by-pixel approach establishes a robust baseline, fire prediction is inherently spatial. The next evolution of this pipeline will transition from tabular Scikit-Learn classifiers to Deep Learning, utilizing Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs) to recognize the physical, 2D shapes of burn scars and terrain slopes.
