# Employee Attrition Predictor

Predicting employee attrition using Random Forest to enable proactive HR intervention before resignations occur.

---

## Executive Summary

This project analyzes IBM's HR dataset of 1,470 employees to identify which individuals are at risk of leaving the company. Using a Random Forest classifier tuned to address class imbalance, the model achieves a ROC-AUC of 0.789 and identifies monthly income, age, and overtime as the strongest predictors of attrition. The output is not limited to a single accuracy figure — the model produces individual risk scores that HR can act on, along with an estimated annual cost saving of approximately ₹29.5 lakhs if attrition is reduced by four percentage points through targeted retention efforts.

---

## Business Problem

Employee turnover is expensive. Industry estimates place the cost of replacing an employee at 50 to 100 percent of their annual salary once recruitment, onboarding, and lost productivity are factored in. In this dataset, attrition runs at 16.1 percent annually across 1,470 employees, equating to roughly 237 exits per year.

The underlying business question is straightforward: can attrition be predicted early enough for HR to intervene, rather than reacting after an employee has already decided to leave.

---

## Dataset

- Source: IBM HR Analytics Employee Attrition Dataset, available on Kaggle (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Size: 1,470 records, 35 original features
- Target variable: Attrition (Yes / No)
- Class distribution: 83.9 percent stayed, 16.1 percent left

The class imbalance is a central consideration throughout this project and is addressed explicitly in the modeling section below.

---

## Methodology

### Exploratory Data Analysis

Before any modeling, several relationships were examined to understand the underlying patterns in the data.

| Area examined | Finding |
|---|---|
| Overtime | Employees working overtime leave at a noticeably higher rate than those who do not |
| Monthly income | Employees who left are concentrated in the lower income bands |
| Tenure | Attrition risk is highest in years one through three, dropping sharply after year five |
| Department | Sales shows the highest attrition rate among all departments |
| Job satisfaction | Low satisfaction scores correlate with eventual departure |

### Data Preparation

Four columns were removed prior to modeling on the grounds that they carried no predictive information: EmployeeCount, StandardHours, and Over18 held a single constant value across all records, and EmployeeNumber functioned purely as a row identifier with no relationship to the outcome. All remaining categorical variables were label encoded. The dataset was split 80/20 into training and test sets using stratified sampling to preserve the original class ratio in both partitions.

### Model Selection and Configuration

A Random Forest classifier was selected over simpler alternatives for three reasons: it handles a mix of numeric and categorical features without requiring scaling, it is comparatively resistant to overfitting when depth is constrained, and it produces feature importance scores as a direct byproduct of training, which has clear business utility beyond the prediction itself.

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 4},
    random_state=42
)
```

The class_weight parameter warrants explanation. With only 16 percent of employees having left, a default model will learn that predicting "stayed" for every employee yields 84 percent accuracy while identifying almost no actual leavers — a technically accurate but practically useless outcome. Setting the weight ratio to 4:1 forces the model to treat a missed leaver as four times more costly than a false alarm, which materially changes what the model optimizes for.

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.789 |
| F1 score, attrition class | 0.37 |
| Precision, attrition class | 0.48 |
| Recall, attrition class | 0.30 |
| Accuracy, overall | 0.84 |
| Test set size | 294 (247 stayed, 47 left) |

A ROC-AUC of 0.789 indicates the model correctly ranks a departing employee above a staying employee in roughly 79 percent of comparisons — a solid result given the inherent difficulty of predicting a relatively rare event from 1,470 records.

Recall of 0.30 for the attrition class means the model identifies three out of every ten employees who actually go on to leave. This is a meaningful improvement over an unweighted baseline, which in early iterations of this model caught barely one in ten. Precision of 0.48 means that of the employees the model flags as likely to leave, close to half genuinely do — an acceptable trade-off given that the cost of a missed departure is generally higher than the cost of an unnecessary HR conversation.

---

## Feature Importance

The features the model relied on most heavily, in order of importance:

1. MonthlyIncome
2. Age
3. OverTime
4. TotalWorkingYears
5. YearsAtCompany
6. DailyRate
7. YearsWithCurrManager
8. StockOptionLevel
9. DistanceFromHome
10. HourlyRate

Monthly income and age emerge as the two dominant factors, followed closely by overtime and career tenure measures. This is consistent with a common attrition pattern: younger employees earlier in their careers, particularly those on lower compensation and working extended hours, represent the highest concentration of flight risk. Total working years and years at the current company reinforce this — the model is effectively learning that career stage matters as much as any single policy factor.

---

## Risk Scoring Output

Beyond a binary prediction, the model generates a probability score for each employee, which is grouped into three tiers for practical HR use.

| Tier | Probability range | Suggested action |
|---|---|---|
| High | Above 60 percent | Direct conversation within the month |
| Medium | 30 to 60 percent | Scheduled check-in within 30 days |
| Low | Below 30 percent | Standard quarterly monitoring |

This framing turns the model from an abstract accuracy metric into an operational tool — a ranked list HR can work through in priority order.

---

## Estimated Business Impact

Using a conservative replacement cost of ₹50,000 per departing employee and the current attrition base of 237 exits annually, a four percentage point reduction in attrition, achieved through intervention on the highest-risk segment, would retain approximately 59 additional employees per year, corresponding to an estimated annual saving of ₹29.5 lakhs. This figure is intentionally conservative and excludes indirect costs such as lost institutional knowledge and reduced team productivity during transition periods.

---

## Recommendations

**Immediate**

Review employees currently flagged as high risk with their respective managers. These conversations should happen before the employee has finalized a decision to leave, not after.

**Short term**

Benchmark compensation for early-career and younger employees against market rates. Monthly income is the single strongest predictor in the model, and this is the segment where pay gaps appear to matter most.

**Medium term**

Audit departments and roles with sustained overtime patterns exceeding three months. Overtime remains a top-five driver, and workload redistribution is a more sustainable fix than after-the-fact retention offers.

**Structural**

Move the risk scoring output into a recurring reporting cycle, ideally integrated into an existing HR dashboard, so that retention becomes a proactive and ongoing process rather than a one-time analysis.

---

## Technical Stack

- Python 3.10
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn for model training and evaluation
- Google Colab as the development environment

---

## Repository Contents

```
employee-attrition-predictor/
    IBM_HR_Attrition_Project.ipynb
    README.md
    chart1_dept_attrition.png
    chart2_overtime_attrition.png
    chart3_income_attrition.png
    chart4_years_attrition.png
    chart5_correlation.png
    chart6_confusion_matrix.png
    chart7_feature_importance.png
```

---

## Limitations and Next Steps

The current model's recall of 0.30, while a substantial improvement over the untuned baseline, still misses a meaningful share of employees who eventually leave. Several extensions would likely improve this further: applying SMOTE to synthetically balance the training data rather than relying on class weighting alone, conducting a more systematic hyperparameter search using cross-validation, and incorporating SHAP values to provide individual-level explanations for each prediction rather than only global feature importance. A production version of this tool would also benefit from a simple interface, such as a Streamlit application, allowing HR staff to query individual employee risk without needing to run the underlying notebook.

---

## Author

Narendra Gopi Krishna Yadlapalli
Data Analytics, Hyderabad

LinkedIn: https://www.linkedin.com/in/narendra-gopi-yadlapalli-121753407
GitHub: https://github.com/narendra732
