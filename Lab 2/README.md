# Lab 2 – Histogram Processing, Histogram Matching and CLAHE

## Aim

To understand and implement **image histogram analysis, histogram matching, customized histogram matching, and Contrast Limited Adaptive Histogram Equalization (CLAHE)** for grayscale images using Python along with OpenCV, NumPy, and Matplotlib.

---

## Objectives

The major objectives of this practical are:

* To study the basic concept and importance of an image histogram.
* To calculate the histogram of grayscale images and represent it graphically.
* To analyze how pixel intensity values are distributed within an image.
* To perform histogram matching between a given source image and a reference image.
* To understand the use of cumulative distribution functions (CDF) in histogram matching.
* To create and use a lookup table for mapping intensity values.
* To construct a mathematical target histogram and perform custom histogram matching.
* To study the working of Contrast Limited Adaptive Histogram Equalization (CLAHE).
* To examine how changes in CLAHE parameters affect image enhancement.
* To visually compare source, reference, matched, and contrast-enhanced images.

---

# Theory

## 1. Image Histogram

An image histogram is a graphical representation showing the **number of pixels corresponding to each intensity value** in an image.

For an 8-bit grayscale image, the possible intensity values are:

**0 to 255**

where:

* `0` corresponds to black.
* `255` corresponds to white.
* Values between 0 and 255 indicate different gray levels.

The histogram indicates the frequency with which every intensity value occurs in the image.

Mathematically, it can be represented as:

**h(r_k) = n_k**

where:

* `r_k` = kth grayscale intensity value
* `n_k` = number of pixels having the intensity `r_k`

By observing a histogram, information about the brightness and contrast characteristics of an image can be obtained.

### Interpretation of Histogram

* A histogram mainly concentrated on the left side indicates a dark image.
* A histogram mainly concentrated on the right side indicates a bright image.
* A narrow distribution generally indicates low contrast.
* A wider distribution generally represents better or higher contrast.

In this practical, OpenCV's `cv2.calcHist()` function is used for calculating the grayscale histograms.

---

# 2. Histogram Normalization

Before performing histogram matching, the histogram is converted into a normalized form so that it represents a probability distribution.

The normalized histogram is given by:

**p(r_k) = n_k / N**

where:

* `n_k` = number of pixels having intensity `r_k`
* `N` = total number of pixels in the image

The source histogram is normalized in the implementation using:

```python
src_hist = src_hist / src_hist.sum()
```

The same normalization procedure is performed for the reference histogram.

Normalization is required before calculating the cumulative distribution function.

---

# 3. Cumulative Distribution Function (CDF)

The Cumulative Distribution Function is obtained by adding the normalized histogram values progressively from the first intensity level up to a particular intensity.

It can be represented as:

**CDF(k) = Σ p(r_j)**

where the summation is performed for intensity values from 0 to `k`.

NumPy's `np.cumsum()` function is used to calculate the CDF:

```python
src_cdf = np.cumsum(src_hist)
ref_cdf = np.cumsum(ref_hist)
```

The CDF plays an important role in histogram matching because it provides the basis for determining the relationship between source and reference intensity levels.

---

# 4. Histogram Matching

Histogram matching, also known as **histogram specification**, is a technique used to modify the intensity distribution of a source image so that it becomes similar to a desired reference distribution.

The main steps involved are:

1. Find the histogram of the source image.
2. Find the histogram of the reference image.
3. Normalize both histograms.
4. Calculate the CDF for each histogram.
5. Compare the source and reference CDF values.
6. Construct an intensity mapping or lookup table.
7. Apply the mapping to the source image.

After this operation, the resulting image has an intensity distribution that approximately follows the distribution of the selected reference image.

---

# 5. Lookup Table for Histogram Matching

A lookup table is used to establish the correspondence between source and reference intensity values.

The table contains 256 entries because an 8-bit grayscale image has 256 possible intensity levels.

It is initialized as:

```python
lookup = np.zeros(256, dtype=np.uint8)
```

For each source intensity, the reference intensity having the closest CDF value is searched.

The mapping is obtained using:

```python
while j < 255 and ref_cdf[j] < src_cdf[i]:
    j += 1

lookup[i] = j
```

Once the lookup table has been generated, it can be applied directly to the source image:

```python
matched = lookup[source]
```

The output obtained after this operation is the histogram-matched image.

---

# 6. Custom Histogram Matching with Analytical Target Histogram

The practical also includes a different form of histogram matching in which an actual reference image is not required.

Instead, a target histogram is mathematically defined using an exponential function:

**target_hist(i) = exp(-i / 50)**

The target distribution is generated for intensity values ranging from `0` to `255`.

After generating the target histogram, it is normalized using:

```python
target_hist = target_hist / target_hist.sum()
```

The corresponding target CDF is obtained with:

```python
target_cdf = np.cumsum(target_hist)
```

The source image is then mapped according to this analytically generated target distribution using a custom histogram matching function.

The generated output is stored in:

```python
stylized
```

This demonstrates that histogram specification can also be performed using a mathematically selected target distribution instead of an existing reference image.

---

# 7. Contrast Limited Adaptive Histogram Equalization (CLAHE)

CLAHE stands for:

**Contrast Limited Adaptive Histogram Equalization**

It is an image enhancement method designed to improve local contrast while controlling excessive enhancement of noise.

Unlike ordinary global histogram equalization, CLAHE divides the image into smaller local regions known as **tiles**.

Histogram equalization is performed independently within each tile.

A contrast limit is also applied so that very high contrast amplification, which may increase noise, can be avoided.

---

## CLAHE Parameters

Two important parameters are considered in this practical.

### 1. Clip Limit

The `clipLimit` determines how much contrast enhancement is allowed during CLAHE processing.

Increasing the clip limit generally permits stronger local contrast enhancement.

For the main implementation, the following value is used:

```python
clipLimit=2.0
```

---

### 2. Tile Grid Size

The `tileGridSize` specifies the arrangement and size of the local regions used during adaptive histogram equalization.

The initial setting used in this practical is:

```python
tileGridSize=(8, 8)
```

Therefore, the image is processed using a grid of local regions.

---

# 8. CLAHE Parameter Sweep

To observe the influence of CLAHE parameters, several values of `clipLimit` and `tileGridSize` are tested.

### Clip Limits

```text
1
2
4
```

and

### Tile Sizes

```text
4
8
16
```

are selected for the experiment.

The total number of combinations is:

**3 × 3 = 9 different CLAHE configurations**

The resulting nine images are arranged in a 3 × 3 subplot grid.

This makes it possible to visually examine how different combinations of `clipLimit` and `tileGridSize` affect the enhancement result.

---

# Software and Libraries Used

The following software and Python modules are used for performing this practical:

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **OS module**

### Libraries Imported

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```

---

# Input Images

Three grayscale images are required for this practical.

### 1. Source 1

```text
source1.jpg
```

This is the main source image used for histogram calculation, histogram matching, custom histogram matching, and CLAHE enhancement.

### 2. Source 2

```text
source2.jpg
```

The second source image is also processed and matched against the reference image to demonstrate histogram matching with another input.

### 3. Reference Image

```text
reference.jpg
```

The reference image provides the desired intensity distribution that the source images are expected to approximate after histogram matching.

---

# Implementation

## Step 1 – Determine the Correct Image Folder

The program first identifies the folder in which the Python script is located.

This is done using:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

Using the script directory helps the program locate the image files correctly even when it is executed from a different working directory.

The complete image paths are constructed using:

```python
os.path.join()
```

---

## Step 2 – Load Images

All three images are opened in grayscale mode using OpenCV:

```python
source1 = cv2.imread(source1_path, cv2.IMREAD_GRAYSCALE)

source2 = cv2.imread(source2_path, cv2.IMREAD_GRAYSCALE)

reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
```

After loading, the program checks whether the images were read successfully.

If an image cannot be found or loaded properly, a `FileNotFoundError` is generated.

---

# Step 3 – Plot Histogram

A separate function named `plot_hist()` is used to calculate and display an image histogram.

The histogram is obtained using:

```python
cv2.calcHist()
```

Since the images are 8-bit grayscale images, the histogram consists of 256 intensity bins.

The X-axis represents:

**Pixel Intensity**

The Y-axis represents:

**Frequency**

Thus, the plotted histogram shows the number of pixels corresponding to each grayscale intensity.

---

# Step 4 – Histogram Matching

A function named:

```python
histogram_matching(source, reference)
```

is implemented to perform histogram matching.

The function follows these operations:

1. Calculate the histogram of the source image.
2. Calculate the histogram of the reference image.
3. Normalize both histograms.
4. Obtain the source CDF.
5. Obtain the reference CDF.
6. Construct an intensity lookup table.
7. Match source intensity values with reference values.
8. Apply the generated lookup table to the source image.

The function is used for both source images:

```python
matched1 = histogram_matching(source1, reference)

matched2 = histogram_matching(source2, reference)
```

As a result, both source images are transformed so that their intensity distributions become closer to that of the reference image.

---

# Step 5 – Generate Analytical Target Histogram

A mathematical target histogram is created using an exponential equation:

```python
target_hist[i] = np.exp(-i / 50)
```

The generated values are normalized and then used to calculate the corresponding CDF.

This provides a predefined target intensity distribution for custom histogram matching.

---

# Step 6 – Custom Histogram Matching

A custom function named:

```python
histogram_match_custom(source, target_cdf)
```

is used to transform the source image according to the analytical target CDF.

The function first calculates the histogram and CDF of the source image and then creates an intensity mapping based on the target distribution.

The resulting stylized image is produced using:

```python
stylized = histogram_match_custom(
    source1,
    target_cdf
)
```

This produces an image whose intensity distribution follows the mathematically specified target more closely.

---

# Step 7 – Apply CLAHE

The CLAHE object is created using:

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
```

The enhancement is then applied to `source1`:

```python
clahe_img = clahe.apply(source1)
```

The resulting image contains enhanced local contrast.

---

# Step 8 – CLAHE Parameter Sweep

To analyze the influence of CLAHE settings, several combinations of clip limits and tile sizes are generated.

The values selected are:

```python
clip_limits = [1, 2, 4]

tile_sizes = [4, 8, 16]
```

Nested loops are used to generate all nine possible combinations.

The parameter combinations are:

```text
Clip=1 Tile=4
Clip=1 Tile=8
Clip=1 Tile=16

Clip=2 Tile=4
Clip=2 Tile=8
Clip=2 Tile=16

Clip=4 Tile=4
Clip=4 Tile=8
Clip=4 Tile=16
```

All nine output images are displayed in a 3 × 3 grid so that the effect of each parameter combination can be compared.

---

# Step 9 – Histogram Comparison

A comparison figure is generated containing both images and their histograms.

### Images

1. Source Image
2. Reference Image
3. Histogram-Matched Image

### Histograms

4. Source Histogram
5. Reference Histogram
6. Matched Histogram

This arrangement makes it easier to compare the visual appearance as well as the intensity distributions before and after histogram matching.

---

# Step 10 – Display Results

Matplotlib is used for displaying histogram graphs and the CLAHE parameter comparison.

OpenCV windows are also used to show:

* Source 1
* Source 2
* Reference
* Matched 1
* Matched 2
* Stylized image
* CLAHE-enhanced image

The program waits for a keyboard input using:

```python
cv2.waitKey(0)
```

After the key is pressed, all OpenCV windows are closed with:

```python
cv2.destroyAllWindows()
```

---

# Algorithm / Procedure

1. Start the program.
2. Find the directory where the Python script is stored.
3. Create paths for Source 1, Source 2, and Reference images.
4. Read all three images in grayscale format.
5. Check whether the images were loaded successfully.
6. Calculate and display the image histograms.
7. Normalize the source and reference histograms.
8. Calculate the CDF of both histograms.
9. Generate a lookup table using the CDF values.
10. Apply the lookup table to Source 1.
11. Repeat the histogram matching operation for Source 2.
12. Create an analytical target histogram using the exponential function.
13. Calculate the CDF of the analytical target.
14. Perform custom histogram matching using the target CDF.
15. Create a CLAHE object using `clipLimit = 2.0` and `tileGridSize = (8,8)`.
16. Apply CLAHE to Source 1.
17. Test different clip limits `1, 2, 4` and tile sizes `4, 8, 16`.
18. Generate nine CLAHE-enhanced outputs.
19. Display the source, reference, and histogram-matched images.
20. Display their respective histograms.
21. Show all processed images using OpenCV.
22. Observe and compare the changes produced by each technique.
23. Stop the program.

---

# Results and Observations

## Histogram Analysis

The histograms of the source and reference images represent the distribution of their grayscale pixel intensities.

From the histogram, the following information can be studied:

* Image brightness
* Image contrast
* Distribution of intensity values
* Most frequently occurring intensity values

---

## Histogram Matching

After applying histogram matching, the intensity distribution of the source image becomes closer to the distribution of the reference image.

Consequently, the appearance of the source image is modified according to the intensity characteristics of the reference image.

The matching operation is carried out for both:

`source1.jpg`

and

`source2.jpg`.

---

## Custom Histogram Matching

The custom matching operation shows that histogram specification can be performed without using a real reference photograph.

Instead, an analytical target distribution can be defined mathematically.

In this practical, the selected target distribution is:

**T(i) = exp(-i / 50)**

The source image is transformed according to this target distribution, producing a stylized output.

---

## CLAHE

CLAHE improves local contrast by dividing the image into smaller regions and processing each region separately.

The parameter sweep demonstrates that changing `clipLimit` and `tileGridSize` changes the appearance and strength of the contrast enhancement.

Testing all nine combinations provides a useful visual comparison of the different CLAHE settings.

---

# Parameter Summary

| Technique                 | Parameters Used                 | Purpose                                                 |
| ------------------------- | ------------------------------- | ------------------------------------------------------- |
| Histogram Matching        | Source + Reference              | Make source intensity distribution similar to reference |
| Custom Histogram Matching | Exponential target distribution | Transform source according to an analytical target      |
| CLAHE                     | Clip Limit = 2.0                | Control the amount of contrast amplification            |
| CLAHE                     | Tile Grid = 8 × 8               | Specify local processing areas                          |
| CLAHE Sweep               | Clip = 1, 2, 4                  | Analyze different contrast limits                       |
| CLAHE Sweep               | Tile = 4, 8, 16                 | Analyze different local region sizes                    |

---

# Output

The following outputs are generated by the practical.

### Histogram Matching

* Histogram-matched version of `source1.jpg`
* Histogram-matched version of `source2.jpg`

### Custom Histogram Matching

* Stylized image generated according to the analytical target histogram

### CLAHE

* CLAHE-enhanced image using `clipLimit=2.0` and `tileGridSize=(8,8)`
* Nine additional CLAHE outputs obtained using different parameter combinations

### Histogram Comparison

A comparison containing:

* Source image
* Reference image
* Matched image
* Source histogram
* Reference histogram
* Matched histogram

---

# Result

Histogram calculation, histogram matching, custom histogram specification, and CLAHE-based contrast enhancement were successfully implemented using Python and OpenCV.

The source images were successfully transformed according to the reference image by using CDF-based intensity mapping. In addition, a mathematically generated target histogram was used to produce a custom stylized result. Different CLAHE parameter combinations were also tested to examine their influence on local contrast enhancement.

---

# Conclusion

In this practical, the concepts of **image histograms, histogram normalization, cumulative distribution functions, histogram matching, histogram specification, and Contrast Limited Adaptive Histogram Equalization (CLAHE)** were studied and implemented.

The histograms of grayscale images were calculated and visualized to understand the distribution of pixel intensities. Histogram matching was performed by comparing the CDF of the source image with the CDF of a reference image and creating a lookup table for mapping intensity values.

Custom histogram matching was also implemented using an exponential target distribution generated mathematically. This demonstrated that a desired intensity distribution can be specified without necessarily requiring a reference image.

Finally, CLAHE was used to improve local contrast while limiting excessive enhancement. Different values of `clipLimit` and `tileGridSize` were tested through a parameter sweep, making it possible to observe their effect on the final enhanced images.

Therefore, the practical successfully demonstrated several histogram-based image enhancement and contrast improvement techniques using **Python, OpenCV, NumPy, and Matplotlib**.
