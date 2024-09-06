# Greenhouse Project

dataset: Advanced_IoT_Dataset.csv (source: Kaggle.com)

I used a dataset (Advanced IoT Agriculture 2024) containing morphometric and quantitative features of plants (presumably from different kind of species) kept in different greenhouses (IoT or Traditional) on different conditions.

In general, the advantage of IoT greenhouses that they can provide optimized environment for the plants. It can be implemented by smart systems for autonomous monitoring of water, temperature, humidity, soil, pH among other parameters.

I aimed the comprehensive comparison of the chlorophyll utilization of the different plants, supposed to be belonged to different species (It was not revealed and detailed in the public source dataset description).

The dataset contains 14 columns, and I used only 5 for the data analysis. 
- Random: It has values R1, R2 and R3 (categorical identifiers), which could represent different plant species.
- Average of chlorophyll in the plant (ACHP): Chlorophyll is vital for photosynthesis, and its measurement can indicate the health and efficiency of the plant in converting light energy into chemical energy.
- Average leaf area of the plant (ALAP): Leaf area is a critical factor in photosynthesis, as it determines the surface area available for light absorption.
- Average number of plant leaves (ANPL): The number of leaves can correlate with the plant's ability to perform photosynthesis and its overall health.
- Class: It indicates the condition or category to which the plant record belongs (SA, SB, SC, TA, TB, TC). This could represent different groups or conditions under which the plants were studied or classified (i.e. IoT vs. traditional). I hypothesized, that SA, SB, SC were smart (IoT), while TA, TB, TC were traditional conditions. 

At first, I carried out exploratory data analysis to discover the dataset and I organized columns, factors, conditions in data frames and columns to prepare them for the statistical tests.

Afterwards, I prepared the descriptive statistics of the prefiltered numerical data: (1) descriptive statistical data, (2) identifying outliers (z-score method), (3) boxplots and histograms, (4) Shapiro-Wilk test (normality test). 'ACHP', 'ALAP', 'ANPL' were non-normally distributed.

Finally, when I had a complete overview of the whole dataset, I started the statistical analysis. 
- Firstly, I hypothesized that there were differences in morphometric and quantitative features of the different ‘species’ (R1, R2, R3). Indeed, significant differences have been proven by Kruskal-Wallis and Mann-Whitney U tests. 
- Secondly, I hypothesized, that the morphometric and quantitative features are significantly different in plant species kept on different conditions. According to the One-way ANOVA results, significant differences have been found. The different conditions have an effect on the chlorophyll content and leaf area of the plants more than on the number of the leaves.

I used Python to perform data analysis (greenhouse_project.ipynb)
