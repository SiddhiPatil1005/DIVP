# Lab 1 – Image Intensity Transformations

## Aim

To understand and implement fundamental image intensity transformation methods such as **Negative Transformation, Power-Law (Gamma) Transformation, and Logarithmic Transformation** using Python and OpenCV, and to study the changes produced in grayscale images.

---

## Objectives

The main objectives of this practical are:

* To learn the basic idea behind image intensity transformations.
* To perform negative transformation on a grayscale image.
* To implement Power-Law or Gamma transformation.
* To perform logarithmic transformation.
* To study how transformation functions change the intensity of individual pixels.
* To understand the transformations through manual pixel-matrix operations.
* To perform the same operations using Python libraries including OpenCV and NumPy.
* To display and compare the original image and the images obtained after transformation.

---

## Theory

Image intensity transformation is an important concept in **Digital Image Processing**. These transformations are used to change pixel intensity values so that the visual quality of an image can be improved or particular image details can be emphasized.

An intensity transformation takes an input intensity value `r` and produces a corresponding output intensity value `s`.

The transformation can generally be written as:

**s = T(r)**

where:

* `r` = intensity value of the input pixel
* `s` = intensity value after transformation
* `T` = transformation function

For an 8-bit grayscale image, the intensity range is normally from **0 to 255**.

In this range:

* `0` indicates black.
* `255` indicates white.
* Values lying between 0 and 255 represent different shades of gray.

Three commonly used intensity transformations are considered in this practical.

---

## 1. Negative Transformation

Negative transformation produces an image in which the intensity levels are inverted.

The mathematical expression for negative transformation is:

**s = (L - 1) - r**

For an 8-bit grayscale image, the equation becomes:

**s = 255 - r**

where:

* `L = 256` represents the total number of intensity levels.
* `r` denotes the original pixel intensity.
* `s` denotes the resulting intensity value.

For example:

| Original Pixel | Negative Pixel |
| -------------: | -------------: |
|              0 |            255 |
|             50 |            205 |
|            100 |            155 |
|            150 |            105 |
|            255 |              0 |

Thus, pixels having low intensity become high intensity, while pixels having high intensity become low intensity.

This technique can be useful when important information is contained in darker portions of an image and needs to be made more visible.

---

## 2. Power-Law (Gamma) Transformation

The Power-Law transformation is commonly referred to as **Gamma transformation**.

Its general mathematical form is:

**s = c × r^γ**

For practical implementation, the input pixel intensity is first converted into the normalized range `[0, 1]`.

The normalization is performed as:

**r_norm = r / 255**

The Power-Law operation is then performed using:

**s_norm = c × (r_norm)^γ**

After the transformation, the normalized result is converted again into the standard intensity range `[0, 255]`.

### Effect of Gamma

The value of `γ` determines how the brightness of the resulting image changes.

* **γ < 1** → produces a brighter image.
* **γ = 1** → approximately maintains the original intensity.
* **γ > 1** → produces a darker image.

For the OpenCV-based implementation, the selected value is:

**γ = 0.5**

Since the Gamma value is less than 1, the resulting image becomes brighter.

A **Lookup Table (LUT)** is used in the OpenCV implementation. It stores the transformed values for the possible 8-bit intensity values and allows the transformation to be applied efficiently.

---

## 3. Logarithmic Transformation

The logarithmic transformation is expressed using:

**s = c × log(1 + r)**

where:

* `r` represents the input intensity.
* `s` represents the transformed intensity.
* `c` is the scaling factor.

The value of the scaling constant is calculated using:

**c = 255 / log(1 + r_max)**

where `r_max` is the highest intensity value present in the image.

The logarithmic operation increases the separation between lower intensity values while reducing the range of higher intensity values.

Because of this property, logarithmic transformation is useful for making details in darker areas more visible.

---

# Software and Libraries Used

The following software and Python libraries are required for this practical:

* **Python**
* **OpenCV (`cv2`)**
* **NumPy**
* **Matplotlib**

### Python Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

# Implementation

The practical is carried out in two major parts:

1. **Performing the transformations manually using a pixel matrix**
2. **Applying the transformations to an image using OpenCV, NumPy and Matplotlib**

---

## Part A – Manual Matrix Implementation

For the first part, an image is represented using a user-provided matrix containing grayscale intensity values.

The program accepts the following information from the user:

* Number of rows
* Number of columns
* Pixel intensity values from 0 to 255
* Gamma value required for the Power-Law transformation

The given matrix is then processed using the three intensity transformation methods.

### 1. Manual Natural Logarithm

A custom function named `manual_ln()` is used to calculate the natural logarithm.

The logarithm is calculated using a Taylor-series-based expression:

**ln(x) = 2 × [z + z³/3 + z⁵/5 + ...]**

where:

**z = (x - 1) / (x + 1)**

Several terms of the series are considered to obtain the logarithmic value without directly depending on Python's built-in logarithm function.

---

### 2. Negative Transformation

For the manual implementation, each pixel is converted into its negative value using:

```python
new_pixel = max_intensity - pixel
```

For an 8-bit grayscale image, `max_intensity` is 255, so the operation becomes:

```text
new_pixel = 255 - pixel
```

This operation is performed individually on all the pixels present in the input matrix.

---

### 3. Power-Law Transformation

Before applying the Power-Law equation, the pixel intensity is normalized.

The normalization step is:

```python
r_norm = pixel / max_intensity
```

The Power-Law operation is then carried out as:

```python
s_norm = c * (r_norm ** gamma)
```

The obtained normalized value is finally converted back to the intensity range `[0, 255]`.

---

### 4. Logarithmic Transformation

For logarithmic transformation, the scaling constant is calculated according to the maximum intensity:

```python
c = max_intensity / manual_ln(1 + max_intensity)
```

The logarithmic operation is then applied to each pixel using:

```python
s = c * manual_ln(1 + pixel)
```

After calculating the new intensity values, they are restricted to the valid range from 0 to 255.

---

# Part B – Image Processing Using OpenCV

In the second part, an actual grayscale image is read using OpenCV.

The image is loaded using:

```python
img = cv2.imread("Lab1/image.jpg", cv2.IMREAD_GRAYSCALE)
```

After loading the image, Negative, Power-Law and Logarithmic transformations are performed on the actual image.

---

## Negative Transformation Using OpenCV

The negative version of the image can be obtained using:

```python
neg_img = cv2.bitwise_not(img)
```

The `bitwise_not()` operation reverses the intensity value of every pixel, thereby generating the negative image.

---

## Power-Law Transformation Using OpenCV

For Gamma transformation, a Lookup Table containing transformed values for all 256 possible grayscale levels is prepared.

The Gamma value used in this practical is:

```python
gamma = 0.5
```

The LUT can be generated using:

```python
lut = np.array(
    [((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]
).astype("uint8")
```

The generated table is applied to the image using:

```python
gamma_img = cv2.LUT(img, lut)
```

The LUT approach improves efficiency because the transformation value for each possible 8-bit intensity is calculated in advance instead of performing the same calculation separately for every pixel.

---

## Logarithmic Transformation Using OpenCV and NumPy

First, the highest intensity value present in the image is determined:

```python
max_pixel_val = float(np.max(img))
```

Using this value, the scaling factor is obtained as:

```python
c = 255 / np.log(1 + max_pixel_val)
```

The logarithmic transformation is then calculated with NumPy:

```python
log_img = np.array(
    c * np.log1p(img.astype(np.float32)),
    dtype=np.uint8
)
```

The function `np.log1p()` is used because it calculates `log(1 + x)` directly and accurately.

---

# Visualization

The transformed images are displayed using Matplotlib so that their differences can be observed easily.

The visualization contains four images:

1. Original Image
2. Negative Transformation
3. Power-Law (Gamma) Transformation
4. Logarithmic Transformation

Four subplot positions are used:

```python
plt.subplot(1, 4, 1)
plt.subplot(1, 4, 2)
plt.subplot(1, 4, 3)
plt.subplot(1, 4, 4)
```

Displaying all four results together makes it easier to compare the changes produced by each transformation with the original grayscale image.

---

# Algorithm / Procedure

1. Begin the program.
2. Define the grayscale image or input pixel matrix.
3. Enter the number of rows, columns and pixel values for the manual implementation.
4. Enter the required Gamma value.
5. Perform the Negative transformation.
6. Perform the Power-Law (Gamma) transformation.
7. Perform the Logarithmic transformation.
8. Display the resulting pixel matrices.
9. Read the grayscale image using OpenCV.
10. Generate the negative image using `cv2.bitwise_not()`.
11. Construct a Lookup Table for the Gamma transformation.
12. Apply the LUT using `cv2.LUT()`.
13. Determine the scaling constant required for logarithmic transformation.
14. Perform the logarithmic operation using NumPy.
15. Display the original and transformed images using Matplotlib.
16. Observe and compare the changes in the resulting images.
17. Stop the program.

---

# Input

The practical requires the following inputs:

* A grayscale image named `image.jpg`.
* Pixel intensity values lying between **0 and 255**.
* A Gamma value entered by the user for the manual Power-Law operation.

For the OpenCV implementation, the image is located at:

```text
Lab1/image.jpg
```

---

# Output

The program generates the following outputs:

### 1. Original Image

The input grayscale image is displayed in its original form without applying any intensity modification.

### 2. Negative Image

The intensity values of the input image are inverted, resulting in a negative representation.

### 3. Power-Law Transformed Image

When `γ = 0.5` is used, the output image becomes brighter and the lower-intensity areas become more prominent.

### 4. Logarithmically Transformed Image

The logarithmic operation increases the representation of lower intensity values and compresses higher intensity values, which can reveal details present in darker portions of the image.

---

# Results and Observations

The three transformation techniques modify the grayscale image in different ways.

| Transformation        | Effect                                                      |
| --------------------- | ----------------------------------------------------------- |
| Negative              | Interchanges low and high intensity values                  |
| Power-Law (`γ = 0.5`) | Increases brightness and emphasizes lower intensity regions |
| Logarithmic           | Expands dark intensity levels while reducing bright levels  |

### Observations

* Negative transformation reverses the complete intensity range of the input image.
* Power-Law transformation allows the brightness of the image to be adjusted by changing Gamma.
* When the Gamma value is below 1, the resulting image becomes brighter.
* Logarithmic transformation is effective for bringing out information present in dark areas.
* The suitable transformation can be selected according to the required image enhancement.
* Implementing the operations using a matrix gives a better understanding of how individual pixels are modified.
* The OpenCV implementation shows how the same mathematical transformations can be efficiently performed on a real grayscale image.

---

# Result

The **Negative, Power-Law (Gamma), and Logarithmic intensity transformations** were successfully performed using both manual pixel-matrix calculations and Python image-processing libraries.

The transformed images were generated successfully and displayed together with the original grayscale image for comparison.

---

# Conclusion

In this practical, three important image intensity transformation techniques — **Negative Transformation, Power-Law (Gamma) Transformation, and Logarithmic Transformation** — were studied and implemented.

Negative transformation was used to invert the intensity levels of the image. Power-Law transformation showed the effect of the Gamma parameter on image brightness, where the selected value of `γ = 0.5` resulted in a brighter image. Logarithmic transformation demonstrated how darker intensity levels can be expanded while higher intensity values are compressed.

The practical also helped in understanding the difference between implementing image transformations manually through pixel matrices and applying them efficiently with **OpenCV, NumPy, and Matplotlib**.

Hence, the practical successfully demonstrated how intensity transformation techniques can be used in **Digital Image Processing** for improving image appearance and analyzing important image details.
