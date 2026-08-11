import numpy as np
import cv2
import matplotlib.pyplot as plt


# ==========================================================
# 1. CUSTOM 2D CORRELATION
# ==========================================================

def correlate2d_custom(image, kernel):

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        image,
        ((ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(kh):
        for j in range(kw):

            output += kernel[i, j] * padded[
                i:i + image.shape[0],
                j:j + image.shape[1]
            ]

    return output


# ==========================================================
# 2. PSNR
# ==========================================================

def calculate_psnr(original, processed):

    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(255 / np.sqrt(mse))


# ==========================================================
# 3. SHARPNESS
# ==========================================================

def calculate_sharpness(image):

    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    response = correlate2d_custom(image, laplacian)

    return np.var(response)


# ==========================================================
# 4. GAUSSIAN NOISE
# ==========================================================

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    result = image.astype(np.float64) + noise

    return np.uint8(np.clip(result, 0, 255))


# ==========================================================
# 5. MOTION BLUR
# ==========================================================

def add_motion_blur(image, length=9):

    kernel = np.zeros((length, length))

    kernel[length // 2, :] = 1 / length

    result = correlate2d_custom(
        image,
        kernel
    )

    return np.uint8(np.clip(result, 0, 255))


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    # ------------------------------------------------------
    # Create clean test image
    # ------------------------------------------------------

    clean_img = np.zeros((100, 100), dtype=np.uint8)

    # Gray road/path
    clean_img[:, 25:75] = 150

    # Bright obstacle
    clean_img[40:60, 40:60] = 255

    # Dark obstacle
    clean_img[10:20, 30:40] = 50


    # ------------------------------------------------------
    # Create degraded images
    # ------------------------------------------------------

    noisy10 = add_gaussian_noise(
        clean_img,
        10
    )

    noisy25 = add_gaussian_noise(
        clean_img,
        25
    )

    blurred = add_motion_blur(
        clean_img,
        9
    )


    # ------------------------------------------------------
    # Baseline
    # ------------------------------------------------------

    print("\n========== BASELINE ==========")

    print(
        "Clean Image Sharpness:",
        round(
            calculate_sharpness(clean_img),
            2
        )
    )


    # ======================================================
    # TASK 1 - AVERAGING FILTER
    # ======================================================

    print("\n========== TASK 1: AVERAGING ==========")

    kernel_sizes = [3, 5, 9]

    averaging_results = {}

    for size in kernel_sizes:

        kernel = np.ones(
            (size, size)
        ) / (size * size)


        result10 = correlate2d_custom(
            noisy10,
            kernel
        )

        result25 = correlate2d_custom(
            noisy25,
            kernel
        )


        result10 = np.uint8(
            np.clip(result10, 0, 255)
        )

        result25 = np.uint8(
            np.clip(result25, 0, 255)
        )


        averaging_results[size] = (
            result10,
            result25
        )


        print(
            f"Sigma 10 | Kernel {size}x{size} | "
            f"PSNR = {calculate_psnr(clean_img, result10):.2f} | "
            f"Sharpness = {calculate_sharpness(result10):.2f}"
        )

        print(
            f"Sigma 25 | Kernel {size}x{size} | "
            f"PSNR = {calculate_psnr(clean_img, result25):.2f} | "
            f"Sharpness = {calculate_sharpness(result25):.2f}"
        )


    # ======================================================
    # TASK 2 - LAPLACIAN SHARPENING
    # ======================================================

    print("\n========== TASK 2: LAPLACIAN ==========")

    lap4 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    lap8 = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])


    response4 = correlate2d_custom(
        blurred,
        lap4
    )

    response8 = correlate2d_custom(
        blurred,
        lap8
    )


    sharp4 = blurred.astype(
        np.float64
    ) - response4

    sharp8 = blurred.astype(
        np.float64
    ) - response8


    sharp4 = np.uint8(
        np.clip(sharp4, 0, 255)
    )

    sharp8 = np.uint8(
        np.clip(sharp8, 0, 255)
    )


    print(
        "4-neighbor PSNR:",
        round(
            calculate_psnr(
                clean_img,
                sharp4
            ),
            2
        )
    )

    print(
        "4-neighbor Sharpness:",
        round(
            calculate_sharpness(sharp4),
            2
        )
    )


    print(
        "8-neighbor PSNR:",
        round(
            calculate_psnr(
                clean_img,
                sharp8
            ),
            2
        )
    )

    print(
        "8-neighbor Sharpness:",
        round(
            calculate_sharpness(sharp8),
            2
        )
    )


    # ======================================================
    # TASK 3 - HIGH BOOST
    # ======================================================

    print("\n========== TASK 3: HIGH BOOST ==========")

    blur_kernel = np.ones(
        (5, 5)
    ) / 25


    smooth = correlate2d_custom(
        noisy10,
        blur_kernel
    )


    mask = (
        noisy10.astype(np.float64)
        - smooth
    )


    k_values = [1, 1.5, 2, 3]

    boost_results = {}


    for k in k_values:

        boosted = (
            noisy10.astype(np.float64)
            + k * mask
        )

        boosted = np.uint8(
            np.clip(boosted, 0, 255)
        )


        boost_results[k] = boosted


        print(
            f"k = {k} | "
            f"PSNR = {calculate_psnr(clean_img, boosted):.2f} | "
            f"Sharpness = {calculate_sharpness(boosted):.2f}"
        )


    # ======================================================
    # DISPLAY 1 - INPUT IMAGES
    # ======================================================

    plt.figure(figsize=(12, 4))

    images = [
        clean_img,
        noisy10,
        noisy25,
        blurred
    ]

    titles = [
        "Clean Image",
        "Gaussian Noise σ=10",
        "Gaussian Noise σ=25",
        "Motion Blur"
    ]


    for i in range(4):

        plt.subplot(1, 4, i + 1)

        plt.imshow(
            images[i],
            cmap="gray"
        )

        plt.title(titles[i])

        plt.axis("off")


    plt.tight_layout()
    plt.show()


    # ======================================================
    # DISPLAY 2 - AVERAGING
    # ======================================================

    plt.figure(figsize=(12, 7))

    for i, size in enumerate(kernel_sizes):

        result10, result25 = averaging_results[size]

        plt.subplot(2, 3, i + 1)

        plt.imshow(
            result10,
            cmap="gray"
        )

        plt.title(
            f"σ=10, {size}x{size}"
        )

        plt.axis("off")


        plt.subplot(2, 3, i + 4)

        plt.imshow(
            result25,
            cmap="gray"
        )

        plt.title(
            f"σ=25, {size}x{size}"
        )

        plt.axis("off")


    plt.suptitle(
        "Averaging Filter"
    )

    plt.tight_layout()
    plt.show()


    # ======================================================
    # DISPLAY 3 - LAPLACIAN
    # ======================================================

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)

    plt.imshow(
        blurred,
        cmap="gray"
    )

    plt.title("Motion Blurred")
    plt.axis("off")


    plt.subplot(1, 3, 2)

    plt.imshow(
        sharp4,
        cmap="gray"
    )

    plt.title("Laplacian 4-Neighbour")
    plt.axis("off")


    plt.subplot(1, 3, 3)

    plt.imshow(
        sharp8,
        cmap="gray"
    )

    plt.title("Laplacian 8-Neighbour")
    plt.axis("off")


    plt.tight_layout()
    plt.show()


    # ======================================================
    # DISPLAY 4 - HIGH BOOST
    # ======================================================

    plt.figure(figsize=(16, 4))

    for i, k in enumerate(k_values):

        plt.subplot(1, 4, i + 1)

        plt.imshow(
            boost_results[k],
            cmap="gray"
        )

        plt.title(
            f"High Boost k={k}"
        )

        plt.axis("off")


    plt.tight_layout()
    plt.show()


# ==========================================================
# START PROGRAM
# ==========================================================

if __name__ == "__main__":
    main()