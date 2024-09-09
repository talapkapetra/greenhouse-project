# Agricultural Data Science Project II

## Table of Contents

- [Data Source](#data-source)
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Tools](#tools)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Statistical Analysis](#statistical-analysis)
- [Results](#results)

## Data Source

I used a dataset available on Kaggle.com

[Advanced_IoT_Dataset.csv]( https://talapkapetra.github.io/greenhouse-project/Advanced_IoT_Dataset.csv)

## Project Overview

Dataset contains morphometric and quantitative features of plants (presumably from different kind of species) kept in different greenhouses (IoT or Traditional) on different conditions.

In general, the advantage of IoT greenhouses that they can provide optimized environment for the plants. It can be implemented by smart systems for autonomous monitoring of water, temperature, humidity, soil, pH among other parameters.

The original dataset contains 14 columns, and I used only 5 for the data analysis.
 - **Random:** It has values R1, R2 and R3 (categorical identifiers), which could represent different plant species.
 - **Average of chlorophyll in the plant (ACHP):** Chlorophyll is vital for photosynthesis, and its measurement can indicate the health and efficiency of the plant in converting light energy into chemical energy.
 - **Average leaf area of the plant (ALAP):** Leaf area is a critical factor in photosynthesis, as it determines the surface area available for light absorption.
 - **Average number of plant leaves (ANPL):** The number of leaves can correlate with the plant's ability to perform photosynthesis and its overall health.
 - **Class:** It indicates the condition or category to which the plant record belongs (SA, SB, SC, TA, TB, TC). This could represent different groups or conditions under which the plants were studied or classified (i.e. IoT vs. traditional). I hypothesized, that SA, SB, SC were smart (IoT), while TA, TB, TC were traditional conditions.

## Objectives

I aimed the comprehensive comparison of the chlorophyll utilization of the different plants, supposed to be belonged to different species (It was not revealed and detailed in the public source dataset description).

## Tools

Python, PyCharm: data cleaning, data preparation, exploratory data analysis, statistics

## Data Preparation

 - I used the following methods to explore the dataset: head(), size, shape, info(), value_counts()
 - Remove duplicated values
 - Organizing the dataset into new dataframes to filter necessary information: 
  - greenhouse_stat: I used only 5 columns from the 14 for this Data Science project: Plant Category=Random), ACHP, ALAP, ANPL, Condition(=Class)
  - R1
  - R2
  - R3
  
## Exploratory Data Analysis

 **Descriptive statistics** 

![descriptive_stats](https://github.com/user-attachments/assets/679151e3-c70f-4043-affb-7766ca04d540)

**Identify Outliers**

I used Z-score method to discover outliers in the datasets. No outliers could be identified in ACHP, ALAP and ANPL.

**Boxplots**

![ACHP](https://github.com/user-attachments/assets/9c7fa7b0-4dc8-44fa-bc5a-6020e5a14df2)

![ALAP](https://github.com/user-attachments/assets/16370f1d-cff2-4527-b777-4b14eb4fa31a)

![ANPL](https://github.com/user-attachments/assets/e0b1efa4-2bd9-4fd7-ad54-a6aa5d7edad2)

**Histograms**

![histograms_greenhouse](https://github.com/user-attachments/assets/6c27f53b-db3c-4b32-82c1-9f37d9752ae0)

**Normality test**

Shapiro-Wilk normality test was used to investigate if the datasets are fit to the Gaussian distribution.

![shapiro_greenhouse](https://github.com/user-attachments/assets/dd30e369-2ad9-42c5-9ec7-1afd3f58afa9)

1. value: t-stat of the test, 2. value: p-value

All of the datasets were non-normally distributed (p < 0.05)

## Statistical Analysis

 - I reorganized the datasets into new dataframes: ACHP_df, ALAP_df, ANPL_df
 - I filled NaN values with mean values.

![boxplots_greenhouse](https://github.com/user-attachments/assets/87a32076-a37d-4ce4-8c47-4053fd425f74)

### Kruskal-Wallis H-test

To compare the whole dataset.

```
Kruskal-Wallis H-test statistic: 5.998208658158546
P-value: 0.04983168117344313
There is a significant difference between the distributions of R1, R2, and R3.
```

### Mann-Whitney U test

I performed pairwise comparisons of the plant categories (R1, R2, R3) in case of all morphometric parameters (ACHP, ALAP, ANPL)

ACHP

```
R1 vs R2: U-statistic = 56971204.0, p-value = 1.1331347670290098e-56
R1 vs R3: U-statistic = 41636201.0, p-value = 4.941714513928499e-102
R2 vs R3: U-statistic = 43681427.0, p-value = 5.995489182060861e-61
```

ALAP

```
R1 vs R2: U-statistic = 55943307.0, p-value = 9.989607968469545e-41
R1 vs R3: U-statistic = 59088291.0, p-value = 4.8441716919346305e-98
R2 vs R3: U-statistic = 47994320.0, p-value = 2.271425719117423e-09
```

ANPL

```
R1 vs R2: U-statistic = 62980828.0, p-value = 4.1334572224423925e-204
R1 vs R3: U-statistic = 63236463.0, p-value = 1.9463838057506312e-212
R2 vs R3: U-statistic = 50757860.0, p-value = 0.4553322219182767
```

### Heatmaps

I created representative heatmaps to visualize differences of morphometric and quantiative features (ACHP, ALAP, ANPL) between the plant categories (R1, R2, R3)

![heatmaps_greenhouse](https://github.com/user-attachments/assets/160a7a46-d180-489d-af75-0e688b7ccd2b)

### Differences between the different conditions

![barcharts_greenhouse](https://github.com/user-attachments/assets/7298ec8c-7b56-44a7-bb05-1e1062f7d0bb)

I created a function to perform Shapiro-Wilk test again for the data sets of conditions.

```python
# H0: The data was drawn from a normal distribution.

def shapiro_test_by_condition(df, col):
    results = []
    conditions = df['Condition'].unique()
    for condition in conditions:
        data = df[df['Condition'] == condition][col]
        stat, p_value = shapiro(data)
        results.append({'Condition': condition, 'W-statistic': stat, 'p-value': p_value})
    return pd.DataFrame(results)

dataframes = [R1, R2, R3]
columns = ['ACHP', 'ALAP', 'ANPL']
titles = ['R1', 'R2', 'R3']

shapiro_results = {}

for df, title in zip(dataframes, titles):
    for col in columns:
        key = f'{title} - {col}'
        shapiro_results[key] = shapiro_test_by_condition(df, col)

for key, result in shapiro_results.items():
    print(f'\nShapiro-Wilk Test Results for {key}')
    print(result)
```

Almost all of the data sets were normally distributed.

```
Shapiro-Wilk Test Results for R1 - ACHP
  Condition  W-statistic   p-value
0        SA     0.998802  0.330430
1        SB     0.999581  0.983705
2        SC     0.998970  0.453220
3        TA     0.999369  0.875895
4        TB     0.998883  0.396184
5        TC     0.998172  0.058251

Shapiro-Wilk Test Results for R1 - ALAP
  Condition  W-statistic   p-value
0        SA     0.998622  0.213720
1        SB     0.999148  0.632874
2        SC     0.998789  0.298868
3        TA     0.999395  0.896206
4        TB     0.998985  0.491017
5        TC     0.999406  0.898166

Shapiro-Wilk Test Results for R1 - ANPL
  Condition  W-statistic   p-value
0        SA     0.998447  0.136711
1        SB     0.999293  0.791019
2        SC     0.998289  0.080455
3        TA     0.998928  0.426201
4        TB     0.999164  0.680255
5        TC     0.997418  0.007285

Shapiro-Wilk Test Results for R2 - ACHP
  Condition  W-statistic   p-value
0        SA     0.999225  0.740038
1        SB     0.999318  0.837448
2        SC     0.998629  0.202757
3        TA     0.999127  0.632356
4        TB     0.998497  0.143468
5        TC     0.999130  0.654874

Shapiro-Wilk Test Results for R2 - ALAP
  Condition  W-statistic   p-value
0        SA     0.997946  0.034578
1        SB     0.998778  0.312058
2        SC     0.999030  0.517418
3        TA     0.999434  0.924238
4        TB     0.998940  0.427898
5        TC     0.999223  0.753326

Shapiro-Wilk Test Results for R2 - ANPL
  Condition  W-statistic   p-value
0        SA     0.998245  0.077669
1        SB     0.999192  0.710340
2        SC     0.999075  0.565066
3        TA     0.998961  0.459804
4        TB     0.999402  0.897183
5        TC     0.998910  0.432361

Shapiro-Wilk Test Results for R3 - ACHP
  Condition  W-statistic   p-value
0        SA     0.998954  0.433945
1        SB     0.998790  0.320933
2        SC     0.998178  0.070718
3        TA     0.999124  0.622871
4        TB     0.999264  0.775860
5        TC     0.999186  0.685509

Shapiro-Wilk Test Results for R3 - ALAP
  Condition  W-statistic   p-value
0        SA     0.998479  0.133120
1        SB     0.998569  0.186685
2        SC     0.999124  0.647127
3        TA     0.999302  0.813225
4        TB     0.999190  0.697905
5        TC     0.999119  0.612397

Shapiro-Wilk Test Results for R3 - ANPL
  Condition  W-statistic   p-value
0        SA     0.999030  0.509794
1        SB     0.998763  0.300939
2        SC     0.999331  0.854671
3        TA     0.999250  0.760163
4        TB     0.998553  0.171332
5        TC     0.999327  0.833225
```

### One-way ANOVA and Tukey's HSD

I created a function to perform One-way ANOVA and Tukey's HSD for a given dataframe and column.

```python
# H0: There are no significant differences in the morphometric and quantiative properties ('ACHP', 'ALAP' or 'ANPL') 
# of plant types (R1, R2 or R3) kept on different conditions (SA, SB, SC, TA, TB, TC).

def anova_and_pairwise(df, col):
    # Perform One-way ANOVA
    groups = [df[df['Condition'] == condition][col] for condition in df['Condition'].unique()]
    anova_result = f_oneway(*groups)
    
    # Tukey's HSD for pairwise comparisons
    tukey = pairwise_tukeyhsd(endog=df[col], groups=df['Condition'], alpha=0.05)
    
    # Formatting p-value to 5 decimal places
    anova_table = pd.DataFrame({
        'F-statistic': [anova_result.statistic],
        'p-value': [round(anova_result.pvalue, 5)]
    })

    # Extract Tukey HSD table and format p-adj values to 5 decimals
    tukey_table = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_table['p-adj'] = tukey_table['p-adj'].round(5)

    return anova_table, tukey_table

dataframes = [R1, R2, R3]
columns = ['ACHP', 'ALAP', 'ANPL']
titles = ['R1', 'R2', 'R3']

anova_results = {}
tukey_results = {}

for df, title in zip(dataframes, titles):
    for col in columns:
        key = f'{title} - {col}'
        anova_results[key], tukey_results[key] = anova_and_pairwise(df, col)

for key in anova_results:
    print(f'\nANOVA Results for {key}')
    print(anova_results[key])
    print(f'\nTukey HSD Results for {key}')
    print(tukey_results[key])
```

## Results
 - ACHP and ALAP: There were significant differences between the different conditions in every cases.
 - ANPL:  Significant differences were not detected between the different conditions in every cases.
 - Different conditions have a significant effect on the chlorophyll content and leaf area of the plants but not on the number of the leaves.
