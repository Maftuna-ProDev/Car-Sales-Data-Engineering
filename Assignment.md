# Big Data and Business Analytics — Assignment Report

---

**Qualification:** Pearson BTEC International Level 3 Extended Diploma in Information Technology  
**Unit:** Unit 10: Big Data and Business Analytics  
**Assignment Title:** Investigate, Explore and Analyze Big Data and Business Analytics

**Student Name:** Baxtiyorov Abdumalik Baxrom o'g'li  
**Group:** 25-102  
**Student ID:** 250076  
**Assessor:** Behzod Qurbonov

**Submission Date:** June 2026

---

## Table of Contents

1. [Task A: Investigation of Big Data and Business Analytics](#task-a-investigation-of-big-data-and-business-analytics)
   - 1.1 Reasons Organizations Analyze Data
   - 1.2 Common Challenges of Data Analysis
   - 1.3 Ethical and Legal Standards
   - 1.4 Types of Data Available
   - 1.5 Data Storage
   - 1.6 Challenges of Big Data Analysis
   - 1.7 Types of Business Analytics
   - 1.8 Benefits and Challenges of Big Data
2. [Task B: Statistical Software Tools and Techniques](#task-b-statistical-software-tools-and-techniques)
   - 2.1 Software Tools Used
   - 2.2 Measures of Central Tendency and Dispersion
   - 2.3 Regression Analysis
   - 2.4 Normal Distribution Analysis
   - 2.5 Visual Analysis
   - 2.6 Frequency and Percentage Analysis
3. [Task C: Data Analysis Results](#task-c-data-analysis-results)
   - 3.1 Data Preparation and Cleaning
   - 3.2 Analysis of 15 Business Questions
   - 3.3 Key Findings
   - 3.4 Conclusions and Recommendations
4. [References](#references)
5. [Appendices](#appendices)

---

## Task A: Investigation of Big Data and Business Analytics

### 1.1 Reasons Organizations Analyze Data

Organizations analyze data for several critical purposes that directly impact their strategic direction and operational effectiveness. Firstly, **strategic planning** relies heavily on data-driven insights. By examining historical sales data, customer demographics, and market trends, organizations can forecast future demand, allocate resources efficiently, and identify growth opportunities. For example, in the car sales industry, analyzing which models and body styles sell most frequently allows dealerships to optimize their inventory and focus marketing efforts on high-demand segments.

Secondly, **improving performance** is a fundamental driver of data analysis. Organizations use Key Performance Indicators (KPIs) derived from data to measure efficiency, customer satisfaction, and financial health. In the context of our car sales dataset, analyzing metrics such as average transaction price, sales volume by region, and customer income brackets enables management to identify underperforming areas and implement targeted improvements.

Thirdly, **benchmarking** products and services against competitors or internal historical performance is a key application. By comparing average prices, customer demographics, and sales volumes across different regions or dealerships, organizations can establish performance standards and identify best practices that can be replicated across the business.

### 1.2 Common Challenges of Data Analysis

Organizations face several significant challenges when implementing data analysis initiatives:

**Cost:** Implementing robust data analytics infrastructure requires substantial investment in hardware, software licensing, cloud storage, and specialized personnel. For small to medium enterprises, these costs can be prohibitive.

**Employee Competency:** There is a well-documented skills gap in the data analytics field. Employees must be proficient in statistical methods, programming languages such as Python or R, and data visualization tools. Without adequate training, organizations cannot fully leverage their data assets.

**Security Issues:** Data breaches and unauthorized access pose significant risks. Customer personal information, financial records, and proprietary business data must be protected through encryption, access controls, and regular security audits.

### 1.3 Ethical and Legal Standards

Data collection, use, and storage are governed by stringent ethical and legal frameworks. The **General Data Protection Regulation (GDPR)** in the European Union sets a global standard for data protection, requiring organizations to obtain explicit consent for data collection, provide transparency about how data is used, and allow individuals to request deletion of their data. Other relevant legislation includes the Data Protection Act in the UK and various consumer privacy laws internationally.

Ethical considerations include ensuring data is not used to discriminate against protected groups, maintaining transparency about algorithmic decision-making, and respecting customer privacy by only collecting data that is necessary for legitimate business purposes. Organizations must implement data governance frameworks that define who can access data, how it can be used, and how long it should be retained.

### 1.4 Types of Data Available

Organizations work with multiple data types:

- **Categorical Data:** Non-numeric data that can be grouped into categories. In our car sales dataset, this includes Gender, Company, Model, Transmission type, Color, Body Style, and Dealer Region.
- **Quantitative Data:** Numeric data that can be measured. This includes Annual Income and Price, both of which can be subjected to mathematical operations and statistical analysis.
- **Internal Data:** Data generated within the organization, such as sales transactions, customer records, and inventory levels. Our car sales dataset is primarily internal transactional data.
- **External Data:** Data obtained from outside the organization, such as market research reports, economic indicators, social media sentiment, and competitor pricing.

### 1.5 Data Storage

Data storage architectures have evolved significantly to handle the volume and variety of modern data:

- **Structured Data:** Data organized in a predefined format, typically stored in relational databases with rows and columns. Our car sales CSV file is an example of structured data, where each row represents a sale and each column represents a specific attribute.
- **Unstructured Data:** Data without a predefined format, such as emails, social media posts, images, and videos. This type of data requires different storage and processing approaches.
- **Data Warehouses:** Centralized repositories that integrate data from multiple sources, optimized for query and analysis rather than transaction processing. Data warehouses enable organizations to perform complex analytical queries across their entire dataset.
- **Data Marts:** Subsets of data warehouses focused on specific business functions or departments. For example, a sales data mart might contain only transaction data relevant to the sales department.

### 1.6 Challenges of Analyzing Big Data

Big data is characterized by the "5 Vs": Volume, Velocity, Variety, Veracity, and Value. The challenges associated with analyzing big data include:

- **Volume:** Managing and processing terabytes or petabytes of data requires distributed computing frameworks such as Apache Hadoop or Spark.
- **Velocity:** Real-time or near-real-time data streams require streaming processing capabilities.
- **Variety:** Integrating structured, semi-structured, and unstructured data from diverse sources presents significant technical challenges.
- **Veracity:** Ensuring data quality, dealing with missing values, and handling inconsistencies requires robust data cleaning and validation processes.

The skills required for big data analysis include proficiency in programming (Python, R, SQL), statistical analysis, machine learning, data visualization, and domain-specific knowledge. Technologies commonly used include pandas, NumPy, scikit-learn, TensorFlow, and cloud platforms such as AWS, Azure, and Google Cloud.

### 1.7 Types of Business Analytics

Organizations employ four main types of business analytics:

1. **Descriptive Analytics:** Answers "What happened?" by summarizing historical data. For example, calculating average car prices, total sales volumes, and customer demographics from our dataset.

2. **Diagnostic Analytics:** Answers "Why did it happen?" by drilling down into data to identify root causes. For example, investigating why certain regions have lower sales volumes or why specific car models underperform.

3. **Predictive Analytics:** Answers "What will happen?" by using statistical models and machine learning to forecast future outcomes. Our regression analysis predicting car prices from customer income and vehicle features is an example of predictive analytics.

4. **Prescriptive Analytics:** Answers "What should we do?" by recommending actions based on predictive insights. For example, optimizing inventory allocation across regions based on predicted demand patterns.

### 1.8 Benefits and Challenges of Big Data

**Benefits:**
- Improved decision-making through data-driven insights
- Enhanced customer understanding and personalization
- Operational efficiency through process optimization
- Competitive advantage through market intelligence
- Innovation in products and services based on data patterns

**Challenges:**
- High implementation and maintenance costs
- Data quality and integration issues
- Privacy and security concerns
- Shortage of skilled data professionals
- Organizational resistance to data-driven culture

---

## Task B: Statistical Software Tools and Techniques

### 2.1 Software Tools Used

For this analysis, I utilized the following Python-based statistical software tools:

- **pandas** (Version 3.0): Used for data loading, cleaning, transformation, and aggregation. Pandas provides DataFrame structures that are ideal for tabular data manipulation.
- **NumPy** (Version 2.4): Used for numerical computations, including statistical calculations and linear algebra operations.
- **Matplotlib** (Version 3.10): Used for creating all visualizations, including histograms, box plots, scatter plots, bar charts, and pie charts.
- **SciPy** (Version 1.17): Used for advanced statistical operations, including the Shapiro-Wilk normality test, t-tests, and probability distribution calculations.

These tools were chosen because they are industry-standard, open-source, and provide comprehensive functionality for data analysis tasks.

### 2.2 Measures of Central Tendency and Dispersion

I calculated both routine and non-routine measures for the numerical variables in the dataset:

**Routine Measures (B.P2):**
- **Mean (Average Price):** $27,425.62 — the arithmetic average of all car prices.
- **Median (Price):** $23,000.00 — the middle value when prices are sorted, which is less affected by outliers than the mean.
- **Mode (Price):** $57,998.50 — the most frequently occurring price point.
- **Range:** $56,798.50 — the difference between the maximum and minimum prices.
- **Variance:** 167,613,672.18 — measures the spread of prices around the mean.
- **Standard Deviation:** $12,946.57 — the square root of variance, providing a more interpretable measure of spread.

**Non-Routine Measures (B.M2):**
- **Skewness:** 0.9679 — indicates the price distribution is positively skewed (tail to the right), meaning there are more lower-priced cars with a few high-priced outliers.
- **Kurtosis:** 0.0092 — indicates the distribution has similar tail thickness to a normal distribution (mesokurtic).
- **Interquartile Range (IQR):** $15,999.00 — the range between the 25th and 75th percentiles, providing a robust measure of spread.
- **Percentiles:** Q1 ($18,001), Q2/Median ($23,000), Q3 ($34,000).

For Annual Income, similar calculations were performed:
- Mean: $804,622.50
- Median: $735,000.00
- Standard Deviation: $626,380.06
- Skewness: 0.6356

### 2.3 Regression Analysis

**Simple Linear Regression (B.P3, B.M3):**

I performed a simple linear regression to model the relationship between Annual Income (predictor) and Car Price (response variable). The resulting equation was:

**Price = 27,240.29 + 0.00023 * Annual Income**

- Pearson correlation coefficient (r): 0.0111
- R-squared: 0.0001
- The very low R-squared value indicates that Annual Income alone is a poor predictor of car price, suggesting that other factors (brand, model, body style) are more significant determinants.

**Multiple Linear Regression (B.M3):**

I extended the analysis to include multiple predictors: Annual Income, Transmission type (Auto/Manual), Engine type (DOHC/OHC), and Body Style. The multiple regression model achieved an R-squared of 0.0102, still relatively low but improved over the simple model. The results show that:

- Automatic transmissions command a premium of approximately $425.47 over manual transmissions
- SUV body styles are associated with lower prices compared to Sedans (the reference category)
- Sedans command the highest premium among body styles (+$518.87 above baseline)

### 2.4 Normal Distribution Analysis

I conducted a normality test on the price data using the Shapiro-Wilk test (B.M2):

- **Shapiro-Wilk statistic:** 0.8926
- **p-value:** 2.07 × 10⁻⁵⁰
- Since the p-value is far below 0.05, we reject the null hypothesis that the data is normally distributed.

This is confirmed by the skewness value (0.9679) and the Q-Q plot, which shows significant deviation from the diagonal line at the tails. The histogram with overlaid normal curve visually demonstrates the right-skewed nature of car prices.

**Z-Score Analysis:** Using the z-score method (|z| > 3), I identified extreme outliers in the price data. After applying IQR-based capping during data cleaning, no extreme outliers remained in the processed dataset, indicating effective outlier handling.

### 2.5 Visual Analysis

I produced 15 visualizations to support the analysis, including:

1. **Price Distribution Histogram** (Q1) — Shows the right-skewed distribution of car prices with KDE overlay
2. **Monthly Sales Trend Line Chart** (Q2) — Reveals seasonal patterns in sales volume
3. **Sales by Region Bar Chart** (Q3) — Horizontal bar chart showing Austin as the leading region
4. **Gender Distribution Pie Chart** (Q4) — Shows 78.6% male, 21.4% female customer base
5. **Income by Region Box Plots** (Q5) — Notched box plots showing income distribution across regions
6. **Income vs Price Scatter Plot** (Q6) — Scatter with regression line showing weak correlation
7. **Average Price by Company Bar Chart** (Q7) — Horizontal bars showing Cadillac as the premium brand
8. **Price by Body Style Box Plots** (Q8) — Shows Sedans commanding highest median prices
9. **Auto vs Manual Price Comparison** (Q9) — Bar chart with error bars and t-test results
10. **Popular Car Colors Bar Chart** (Q10) — Pale White dominates at 47.1%
11. **Body Style × Transmission Heatmap** (Q11) — Cross-tabulation with color intensity
12. **Multiple Regression Diagnostic Plots** (Q12) — Predicted vs actual and residual histogram
13. **Outlier Detection Scatter Plot** (Q13) — Z-score based outlier identification
14. **Average Price by Dealer Bar Chart** (Q14) — Top dealers ranked by average price
15. **Normality Q-Q Plot** (Q15) — Theoretical vs sample quantiles for normality assessment

### 2.6 Frequency and Percentage Analysis

I computed frequency distributions and percentages for all categorical variables:

- **Gender:** Male — 18,798 (78.6%), Female — 5,108 (21.4%)
- **Transmission:** Automatic — 12,571 (52.6%), Manual — 11,335 (47.4%)
- **Color:** Pale White — 11,256 (47.1%), Black — 7,857 (32.9%), Red — 4,793 (20.0%)
- **Body Style:** SUV — 6,374 (26.7%), Hatchback — 6,128 (25.6%), Sedan — 4,488 (18.8%), Passenger — 3,945 (16.5%), Hardtop — 2,971 (12.4%)
- **Dealer Region:** Austin — 4,135 (17.3%), Janesville — 3,821 (16.0%), Scottsdale — 3,433 (14.4%), others approximately 13% each.

---

## Task C: Data Analysis Results

### 3.1 Data Preparation and Cleaning

The raw dataset contained 23,906 records across 16 columns. The data cleaning process involved:

1. **Column Name Standardization:** Removed spaces and special characters from column names (e.g., "Price ($)" became "Price", "Dealer_No " became "Dealer_No").
2. **Date Parsing:** Converted the Date column from string format to datetime objects for time series analysis.
3. **Encoding Fix:** Removed "Â" encoding artifacts from the Engine column.
4. **Numeric Conversion:** Converted Annual Income and Price columns to numeric types, coercing any invalid values to NaN.
5. **Missing Value Handling:** Dropped 0 rows with missing income or price values, indicating good data completeness.
6. **Duplicate Removal:** Removed 0 duplicate Car_id records.
7. **Outlier Capping:** Applied IQR-based capping (1.5× IQR rule) to Price and Annual Income, capping 1,449 price outliers and 816 income outliers to preserve data integrity.

The cleaned dataset was saved to `outputs/cleaned_data.csv` for further analysis.

### 3.2 Analysis of 15 Business Questions

The analysis answered 15 specific business questions relevant to the sales organization:

**Q1: Price Distribution** — Car prices range from $1,200 to $57,998.50 with a mean of $27,425.62. The distribution is right-skewed, indicating that most cars are priced below $34,000.

**Q2: Monthly Sales Trend** — Sales show clear seasonal patterns with peaks in December 2023 (1,921 sales) and troughs in January 2022 (315 sales), suggesting strong end-of-year purchasing behavior.

**Q3: Sales by Region** — Austin leads with 4,135 sales (17.3%), followed by Janesville (3,821) and Scottsdale (3,433). Sales are relatively balanced across the remaining regions.

**Q4: Gender Distribution** — Male customers account for 78.6% of purchases, suggesting a need for targeted marketing strategies to increase female customer engagement.

**Q5: Income by Region** — Median annual income is relatively consistent across regions ($720,000-$750,000), though Pasco shows slightly higher average incomes.

**Q6: Income vs Price Correlation** — A very weak positive correlation (r = 0.011, R² = 0.0001) exists between income and price, suggesting that price is not primarily determined by customer income.

**Q7: Average Price by Company** — Cadillac commands the highest average price at $37,557.37, followed by Saab ($33,670.28) and Buick ($32,929.48). Ford, Toyota, and Honda are mid-range.

**Q8: Price by Body Style** — Sedans have the highest median price ($25,000), while SUVs have the lowest median ($22,350). Hardtops show the widest price variation.

**Q9: Automatic vs Manual** — Automatic cars are significantly more expensive on average ($27,853 vs $26,951, p = 8.0×10⁻⁸), confirming a statistically significant price premium.

**Q10: Popular Colors** — Pale White dominates with 47.1% of sales, followed by Black (32.9%) and Red (20.0%). These three colors account for all sales in the dataset.

**Q11: Body Style × Transmission** — SUVs are the only body style where manual transmissions (3,288) outnumber automatic (3,086). Hatchbacks show the strongest preference for automatic transmissions.

**Q12: Multiple Regression** — The model explains only 1.02% of price variance (R² = 0.0102). Body style and transmission type have more influence than income.

**Q13: Price Outliers** — After IQR capping, no extreme outliers (|z| > 3) remain, though the data is still right-skewed.

**Q14: Dealer Price Ranking** — U-Haul Co. has the highest average price ($28,053.76), while Star Enterprises Inc. has the lowest among top dealers ($27,382.85). The variation between dealers is relatively small.

**Q15: Normality Test** — Price data is not normally distributed (Shapiro-Wilk p < 0.001), confirming the need for non-parametric statistical methods in some analyses.

### 3.3 Key Findings

1. **Total revenue** generated from 23,906 car sales was **$655,636,900.50**.
2. The **average car price** is $27,425.62, with a median of $23,000.00.
3. **SUV** and **Hatchback** body styles together account for 52.3% of all sales.
4. **Cadillac** is the premium brand by average price ($37,557).
5. **Austin, Texas** is the largest market (17.3% of sales).
6. **Automatic transmissions** command a statistically significant price premium.
7. **Pale White** is overwhelmingly the most popular color choice.
8. Customer income is **not a strong predictor** of car purchase price, suggesting other factors (brand preference, financing options, model features) are more influential.

### 3.4 Conclusions and Recommendations

**Conclusions:**

This analysis demonstrates the power of big data analytics in a sales organization context. By applying descriptive, diagnostic, and predictive analytical techniques to car sales data, we have uncovered valuable insights about customer behavior, market dynamics, and pricing patterns. The data cleaning process ensured high data quality, while the application of both routine and non-routine statistical operations provided a comprehensive understanding of the dataset.

The relatively low explanatory power of the regression models suggests that car pricing is influenced by a complex combination of factors beyond income and basic vehicle attributes, including brand perception, financing terms, negotiation outcomes, and market conditions.

**Recommendations for the Sales Organization:**

1. **Inventory Optimization:** Prioritize stocking SUVs and Hatchbacks, which together account for over half of all sales. Within each body style, focus on the most popular models and price points.

2. **Regional Strategy:** Allocate more inventory and marketing resources to the Austin region, which generates the highest sales volume. Investigate why other regions underperform and implement targeted improvement strategies.

3. **Customer Segmentation:** Develop targeted marketing campaigns to increase female customer engagement (currently only 21.4%). The income data can be used to segment customers into appropriate financing and vehicle tiers.

4. **Pricing Strategy:** The weak correlation between income and price suggests that emotional and brand-driven factors heavily influence purchasing decisions. Marketing should emphasize brand value, features, and lifestyle benefits rather than price alone.

5. **Color Inventory:** Given that 47.1% of customers choose Pale White, dealerships should ensure adequate stock of this color, particularly for popular models.

6. **Predictive Analytics:** Implement real-time data collection systems to enable more sophisticated predictive and prescriptive analytics. Machine learning models incorporating additional features (financing terms, trade-in values, customer history) could substantially improve price prediction accuracy.

7. **Benchmarking:** Use the dealer performance rankings and regional comparisons to establish performance benchmarks and share best practices across the organization.

---

## References

Marr, B. (2015) *Big Data: Using Smart Big Data, Analytics and Metrics to Make Better Decisions and Improve Performance*. Chichester: John Wiley & Sons, Ltd.

Cody, I.D. (2016) *Data Analytics: Practical Data Analysis and Statistical Guide to Transform and Evolve Any Business*. North Charleston: Create Space Independent Publishing Platform.

McKinney, W. (2022) *Python for Data Analysis*. 3rd edn. Sebastopol: O'Reilly Media.

Wickham, H. and Grolemund, G. (2017) *R for Data Science*. Sebastopol: O'Reilly Media.

Provost, F. and Fawcett, T. (2013) *Data Science for Business*. Sebastopol: O'Reilly Media.

---

## Appendices

### Appendix A: Data Cleaning Code (Screenshot Reference)

The data cleaning function `load_and_clean_data()` in `src/analysis.py` handles:
- CSV loading with encoding handling
- Column name standardization
- Date parsing to datetime
- String encoding cleanup
- Numeric type conversion
- Missing value handling
- Duplicate removal
- IQR-based outlier capping

### Appendix B: Statistical Calculations Summary

| Metric | Price ($) | Annual Income ($) |
|--------|-----------|-------------------|
| Mean | 27,425.62 | 804,622.50 |
| Median | 23,000.00 | 735,000.00 |
| Std Dev | 12,946.57 | 626,380.06 |
| Skewness | 0.9679 | 0.6356 |
| Kurtosis | 0.0092 | -0.0919 |
| IQR | 15,999.00 | 789,750.00 |

### Appendix C: List of Figures

| Figure | File Name | Description |
|--------|-----------|-------------|
| 1 | q01_price_distribution.png | Histogram of car prices with KDE |
| 2 | q02_monthly_sales_trend.png | Monthly sales over time |
| 3 | q03_sales_by_region.png | Sales volume by dealer region |
| 4 | q04_gender_distribution.png | Customer gender pie chart |
| 5 | q05_income_by_region.png | Income distribution by region |
| 6 | q06_income_vs_price.png | Scatter with regression line |
| 7 | q07_avg_price_by_company.png | Average price by company |
| 8 | q08_price_by_body_style.png | Price distribution by body style |
| 9 | q09_auto_vs_manual_price.png | Auto vs manual price comparison |
| 10 | q10_popular_colors.png | Most popular car colors |
| 11 | q11_style_transmission_heatmap.png | Body style × transmission heatmap |
| 12 | q12_multiple_regression.png | Multiple regression diagnostics |
| 13 | q13_outliers_zscore.png | Z-score outlier detection |
| 14 | q14_avg_price_by_dealer.png | Average price by dealer |
| 15 | q15_normality_test.png | Q-Q plot and normality test |

---

*End of Report*

*Word Count: Approximately 2,200 words*
