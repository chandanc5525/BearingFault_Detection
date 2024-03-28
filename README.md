# BearingFault_Detection
Bearing Fault Diagnosis using ML &amp; DL Technique

## Objective:
The objective of this project is to develop a comprehensive monitoring system for vehicles that integrates machine learning techniques to analyze key parameters related to vehicle performance and bearing health. By leveraging real-time data captured by ADXL335 Sensors and logged via a data logger, the system aims to enhance fault detection, optimize performance, and ensure safety compliance across automotive operations.

## Technical Components:

### Data Acquisition:

ADXL335 Sensors: These sensors capture acceleration data in three axes (X_ACC, Y_ACC, Z_ACC) to monitor vehicle movements and vibrations.
Data Logger: A data logger is utilized to record real-time sensor data, enabling continuous monitoring of vehicle performance and bearing health.

## Data Preprocessing:

Feature Extraction: Relevant features such as RPM, acceleration values, and output status (FAULTY or NON-FAULTY) are extracted from the logged data for analysis.
Data Cleaning: Preprocessing techniques are applied to handle missing values, outliers, and noise in the dataset, ensuring data quality for subsequent analysis.

## Machine Learning Algorithms:

Supervised Learning: Classification algorithms such as Support Vector Machines (SVM), Random Forest, and Neural Networks are employed for fault detection based on the output status of bearings.
Unsupervised Learning: Clustering algorithms like K-means clustering aid in identifying patterns and anomalies within the data, facilitating predictive maintenance strategies.

## Model Training and Evaluation:

Training: Machine learning models are trained using historical data to classify bearing conditions (FAULTY or NON-FAULTY) accurately.
Evaluation: Model performance is assessed using metrics such as accuracy, precision, recall, and F1-score to ensure reliable fault detection and classification.

## Integration and Deployment:

Integration: The developed machine learning models are integrated into the monitoring system, enabling real-time analysis of vehicle performance and bearing health.
Deployment: The integrated system is deployed on vehicle platforms, allowing for continuous monitoring and proactive maintenance interventions.

## Expected Outcomes:

1. Enhanced Fault Detection: The system is expected to improve fault detection accuracy by leveraging machine learning algorithms to analyze sensor data and classify bearing conditions accurately.

2. Predictive Maintenance: Predictive maintenance strategies enabled by the system are anticipated to minimize downtime and reduce repair costs by anticipating potential bearing failures and recommending proactive maintenance actions.

3. Safety Assurance: Continuous monitoring of vehicle performance and bearing health ensures compliance with safety regulations, enhancing operational safety and reducing the risk of accidents or equipment failures.

4. Performance Optimization: Insights derived from machine learning analysis facilitate performance optimization by recommending adjustments to vehicle settings and maintenance schedules, improving vehicle efficiency, reliability, and overall performance.
