# 🕵️🧪 Null & Alternative Hypothesis: Easy read for beginners 👀 (Est. 12 mins)

## 🧠 **Understanding Hypothesis Testing**
Hypothesis testing is a statistical method used to **make inferences or draw conclusions about a population based on a sample of data**✨. It involves setting up two competing hypotheses, the null hypothesis (H0) and the alternative hypothesis (H1), and using sample data to determine which hypothesis is more likely to be true.

### 0️⃣**H0: Null Hypothesis**
The null hypothesis, denoted as H0, is a statement that there is no significant difference or effect. It represents a default assumption or a statement of no effect. In other words, the null hypothesis assumes that any observed differences or effects in the data are due to random chance or sampling variability.

### 1️⃣**H1: Alternative Hypothesis**
The alternative hypothesis, denoted as H1, is a statement that contradicts the null hypothesis. It suggests that there is a significant difference or effect in the data, beyond what would be expected by random chance. The alternative hypothesis is what researchers often hope to demonstrate or find evidence for.

### 🎯**Objective of Hypothesis Testing**
The goal is to assess whether there is enough evidence in the data to reject the null hypothesis in favor of the alternative hypothesis. This assessment is typically done using statistical tests, p-values, and significance levels.

### 🔄**Error types:**
Type I error and Type II error are terms used in hypothesis testing to describe the kinds of errors that can occur when making a decision about a null hypothesis.

1️⃣ **Type I Error (False Postive):**
- **Definitiong:** Rejecting a true null hypothesis
- **Symbol:** &alpha; (alpha)
- **Explanation:** This error occurs when you conclude that there is a significant effect or difference when, in reality, there is none. It's like a "false alarm" or a "false positive" result.
- **Example:** Suppose a medical test for a disease has a 5% false-positive rate. 
>If a healthy person is incorrectly diagnosed as having the disease, it's a Type I error.

2️⃣ **Type 2 Error (False Negative):**
- **Definitiong:** Failing to reject a false null hypothesis.
- **Symbol:** &beta; (beta)
- **Explanation:** This error occurs when you fail to detect a significant effect or difference that actually exists. It's like missing an important finding or making a "miss."
- **Example:** Using the same medical test, a Type II error would occur.
>If a person with the disease is incorrectly classified as healthy.

📢**Simply:**
- Type I Error (False Positive): Incorrectly rejecting a true null hypothesis.
- Type II Error (False Negative): Failing to reject a false null hypothesis.

## 🛠️**Example:**

### 1️⃣**Step 1: Formulate Hypothesis**
0️⃣ **Null Hypothesis (H0):** The mean height of plants with the new fertilizer is equal to the mean height without the fertilizer.

>H0 : mean height of plants with fertilizer = mean height of plants without fertilizer

1️⃣ **Alternative Hypothesis (H1):** The mean height of plants with the new fertilizer is different from the mean height without the fertilizer.

>H1: mean height wit fertilizer &#8800; mean height without fertilizer

### 2️⃣**Step 2: Choose a Significance Level (&alpha;)**
Let's choose a **common significance level of 
0.05**, indicating that we are willing to accept a 5% chance of making a Type I error (incorrectly rejecting a true null hypothesis).

### 3️⃣**Step 3: Collect and Analyze Data**
Collect data on the height of plants with and without the new fertilizer. Let's say we have two datasets representing the heights of 30 plants in each group.

### 4️⃣**Step 4: Calculate a Test Statistic**
Choose an appropriate statistical test based on the data and hypotheses. Let's assume we use a two-sample t-test to compare the means.

### 5️⃣**Step 5: Determine the Critical Region (Rejection Region)**
Based on the chosen significance level and the degrees of freedom, find the critical values for the t-test. This defines the range of values that would lead to rejecting the null hypothesis.

### 6️⃣**Step 6: Make a Decision**
Calculate the test statistic from the data and compare it to the critical values. If the test statistic falls into the critical region, reject the null hypothesis. If not, fail to reject the null hypothesis.

### 7️⃣**Step 7: Draw a Conclusion**
Based on the decision, draw a conclusion about the null hypothesis and make an inference about the effect of the new fertilizer on plant height.

### 8️⃣**Step 8: Calculate a p-value**
Optionally, calculate the p-value associated with the test statistic. A lower p-value provides additional information about the strength of evidence against the null hypothesis.

> In summary, hypothesis testing helps you make a scientific decision based on data, allowing you to assess whether the observed differences are likely due to a real effect or just random variability.

👾**Code:**
```
import numpy as np
from scipy import stats

# Step 3: Collect and Analyze Data
with_fertilizer = np.array([25, 26, 28, 30, 32, 27, 29, 31, 33, 28, 30, 31, 32, 29, 30, 28, 27, 31, 32, 30, 31, 29, 28, 30, 31, 27, 28, 29, 30, 31])
without_fertilizer = np.array([22, 23, 24, 25, 26, 23, 22, 25, 24, 22, 23, 24, 25, 26, 23, 22, 24, 23, 25, 26, 22, 23, 24, 25, 26, 22, 23, 24, 25, 26])

# Step 4: Calculate a Test Statistic
t_statistic, p_value = stats.ttest_ind(with_fertilizer, without_fertilizer)

# Step 5: Determine the Critical Region (Rejection Region)
alpha = 0.05
critical_value = stats.t.ppf(1 - alpha / 2, len(with_fertilizer) + len(without_fertilizer) - 2)

# Step 6: Make a Decision
if np.abs(t_statistic) > critical_value:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# Step 7: Draw a Conclusion
if p_value < alpha:
    print("Based on the p-value, reject the null hypothesis")
else:
    print("Based on the p-value, fail to reject the null hypothesis")

# Step 8: Calculate a p-value (Optional)
print("p-value:", p_value)
```

## 📢 **Conclusion:**
Imagine hypothesis testing like being a detective🕵️ trying to solve a mystery. You have a theory, called a hypothesis, about what might be happening. Hypothesis testing helps you figure out if your theory is right or not. It's like being a scientific detective, using data and evidence to decide if your idea is likely true or just a guess. For example, if you believe a new plant fertilizer makes plants grow taller, hypothesis testing lets you check if the data supports this idea or if it's just a coincidence. It's a way to be sure your conclusions are solid and not based on luck. In simple terms, it's a tool to make smarter and more confident decisions in all sorts of areas.