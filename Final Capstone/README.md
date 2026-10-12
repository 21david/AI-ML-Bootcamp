# Banking operations observability

This project takes the bank operations data and builds models to forecast metrics, flag critical incidents, and help explain issues from logs and runbooks.

## Files

- `notebooks/final-capstone-task-1.ipynb` — data cleaning and exploration
- `notebooks/final-capstone-task-2.ipynb` — forecasting, classification, and anomaly detection
- `notebooks/final-capstone-task-3.ipynb` — log clustering, knowledge base, RAG, and an AI assistant
- `data/` — CSVs used by the notebooks
- `models/` — saved models from Task 2
- `knowledge_base/` — troubleshooting documents used in Task 3

## Data sources

- `infrastructure_metrics.csv`, `infrastructure_metrics_train.csv`, `infrastructure_metrics_test.csv` -> Minute-level metrics about server, transaction, success-rate, and processing-time metrics
- `incident_reports.csv` -> Hour-level incidents with severity, critical labels, and related KPIs
- `application_logs.csv` -> Application logs with INFO, WARNING, and ERROR levels and descriptions
- `transactions_fraud_train.csv`, `transactions_fraud_test.csv` -> Transactions labeled as fraud or not
- `credit_applications_train.csv`, `credit_applications_test.csv` -> Customer loan applications
- `customer_demographics.csv` -> Customer age, gender, income level, city, and account details

## Task 1

I loaded the main tables (`infrastructure_metrics`, `incident_reports`, `application_logs`, transactions, credit applications, and customer demographics), checked missing values and duplicates, set data types, and explored patterns in the data including distributions, correlations between features, and patterns.

Main findings:

- Server metrics (CPU, memory, DB pool, network) go up with transaction volume during 9am–6pm on weekdays, which cause transaction processing times to slow down.
- NEFT/IMPS/RTGS success rates stay in a narrow range and do not follow that pattern.
- In `incident_reports`, every critical incident is `Resource_Exhaustion`. CPU is at least 85.4% in all of these, memory at least 80.8%, and NEFT success drops. A CPU threshold between 76% and 85% separates critical from non-critical rows.

## Task 2

- **Forecasting:** Random Forest models predict NEFT success rate, average transaction processing time (used as the latency target), and CPU utilization 60 minutes ahead. They beat a naive baseline on RMSE and MAE. NEFT success is mostly noise, so the forecast stays near the average.
- **Classification:** A Random Forest predicts `is_critical` from KPIs. Accuracy and F1 are 1.0 on the test set because of the CPU gap discovered in Task 1.
- **Anomaly detection:** Isolation Forest on the 9 KPIs flagged 161 of 10,080 minutes (1.6%). It had recall 1.0 and F1-score 0.945 against labelled incidents.


## Task 3

- **Log processing and clustering:** Logs are first cleaned and turned into TF-IDF vectors, then they are clustered into infrastructure issues vs business KPI issues through KMeans.
- **Knowledge base:** A knowledge base is built from historical incidents and the incident knowledge base entries for the RAG-enabled assistant to use.
- **RAG and AI assistant:** A RAG pipeline retrieves the most relevant documents for a query. A LangChain assistant answers questions from that context and writes root cause summaries and recommended next steps, enabling banks to resolve issues more quickly.

## How to run

Open the notebooks and run them in order.

The RAG part of task 3 requires an OpenAI API key which should be placed in `environment/.env`. 

The `ask` function can be tested with different prompts to see what the specialized banking assistant responds.

The `summarize_incident` function can take in any row that has the same columns as `incident_reports` table to see what the banking assistant is able to diagnose and recommend.
