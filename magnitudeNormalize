import math
# To read the image and to take the DFT result of image that is used as input parameter of my implementation.
import cv2
# Used to find the required input DFT result for implemented magnitude spectrum function.
import numpy as np
# To display the image which is the result of the implemented functions.
import matplotlib.pyplot as plt

# Normalize image function provides to normalize an image by setting the mean value
# to 0 and standard deviation to 1. It takes the read image as the input parameter.
# If the given image is colourful it gives a colourful result. Otherwise, gives grayscale result.
def normalize_image(image):
    summation_pixels = 0
    number_pixels = 0
    sum_squared_diff = 0

    # If the image is a grayscale it uses first part.
    # It uses each pixel value one by one.
    if len(image.shape) == 2:
        # Mean of pixels of image is calculated by scanning each pixel one by one using for loops.
        for row in image:
            for pixel in row:
                summation_pixels += pixel
                number_pixels += 1
        mean = summation_pixels / number_pixels

        # Standard deviation of the image is estimated by using calculated mean value of the image and each pixel value.
        for row in image:
            for pixel in row:
                sum_squared_diff += (pixel - mean) ** 2
        # Standard deviation value of image is calculated.
        standard_deviation = math.sqrt(sum_squared_diff / number_pixels)

        # Normalizing of image using standard deviation and mean values of the image.
        final_normalized_image = []
        # First for loop scans each row of the image.
        for row in image:
            normalized_row = []
            # Second for loop provides to normalize each pixel one by one.
            for pixel in row:
                # Each pixel normalized using pixel value, mean and standard deviation.
                normalized_pixel = (pixel - mean) / standard_deviation
                # Normalized pixels added to the normalized rows.
                normalized_row.append(normalized_pixel)
            # Normalized rows added to the normalized image.
            final_normalized_image.append(normalized_row)
        # Returns normalized version of the image.
        return final_normalized_image

    # If the image is a colourful image, it uses second part.
    # It should normalize each pixel value by considering them with value of each  channel  R, G, B.
    # Because each pixel consists of three channels.
    elif len(image.shape) == 3:
        # Mean of pixels for each channel of image is calculated by scanning each pixel one by one using for loops.
        for row in image:
            for pixel in row:
                summation_pixels += sum(pixel)
                # Each pixel has 3 channels (R, G, B). That's why number of pixels are increased by one for each.
                number_pixels += 3
        # Mean value of each channel is calculated.
        mean = summation_pixels / number_pixels

        # The sum of squared differences from the mean is calculated for each channel.
        for row in image:
            for pixel in row:
                for channel in pixel:
                    sum_squared_diff += (channel - mean) ** 2
        # Standard deviation of the image is estimated by using calculated mean value of the image and found difference.
        standard_deviation = math.sqrt(sum_squared_diff / number_pixels)

        # Normalizing of each channel of image is realized using standard deviation and mean values of the image.
        normalized_image = []
        for row in image:
            normalized_row = []
            for pixel in row:
                # Each pixel normalized using channel, mean and standard deviation.
                normalized_pixel = [(channel - mean) / standard_deviation for channel in pixel]
                # Normalized pixels added to the normalized rows.
                normalized_row.append(normalized_pixel)
            # Normalized rows added to the normalized image.
            normalized_image.append(normalized_row)

    # Normalized image is scaled using min and max to 0-1 range for getting the visualization properly.
    min_value = min(min(channel for pixel in row for channel in pixel) for row in normalized_image)
    max_value = max(max(channel for pixel in row for channel in pixel) for row in normalized_image)

    # Stores scaled normalized image.
    final_normalized_image = []
    for row in normalized_image:
        scaled_row = []
        for pixel in row:
            # Each pixel value is set to 0 by subtracting its minimum value and
            # then scaled between 0 and 1 by dividing by (max_value - min_value).
            scaled_pixel = [(channel - min_value) / (max_value - min_value) for channel in pixel]
            # Scaled pixels added to the scaled rows.
            scaled_row.append(scaled_pixel)
        # Scaled rows added to the scaled image.
        final_normalized_image.append(scaled_row)

    return final_normalized_image


# Magnitude spectrum function provides to find magnitude spectrum of an image.
# It takes DFT result that is a 2D complex matrix of the image as the input parameter.
# Firstly, it shifts the DFT result then, finds magnitude.
# Then, it scales the magnitude result by log transform to get the visual of magnitude spectrum.
def magnitude_spectrum(dft_result):
    # Create an empty matrix for shifting.
    shifted_dft = []
    n = len(dft_result)
    m = len(dft_result[0])
    # Shift each element one by one.
    for i in range(n):
        shifted_row = []
        for j in range(m):
            # The new position of each element is calculated and added to the shifted matrix.
            shifted_row.append(dft_result[(i + n // 2) % n][(j + m // 2) % m])
        shifted_dft.append(shifted_row)

    # It holds the values inside the magnitude list.
    magnitude_result = []
    # It scans each row of shifted_dft one by one.
    for row in shifted_dft:
        magnitude_row = []
        # For loop provides to find the magnitude values of each complex numbers.
        for complex_number in row:
            # Calculates the magnitude of complex numbers.
            magnitude = math.sqrt(complex_number[0] ** 2 + complex_number[1] ** 2)
            # Each calculated magnitude value is appended to the list.
            magnitude_row.append(magnitude)
        # Magnitude values of rows are appended to the magnitude list.
        magnitude_result.append(magnitude_row)

        # Logarithmic transform provides to scale the calculated magnitude spectrum result to be able to display it.
        # Converting the magnitude spectrum result logarithmically.
        magnitude_transformed_result = []
        for row in magnitude_result:
            transformed_row = []
            for mag in row:
                # Logarithmic conversion is provided.
                transformed_mag = 20 * math.log(mag)
                transformed_row.append(transformed_mag)
            magnitude_transformed_result.append(transformed_row)

    return magnitude_transformed_result

# Path of the image is given.
image_path = 'cat.jpg'

# Image is opened using image path.
image = cv2.imread(image_path)

# The results of the normalize image and magnitude spectrum functions are displayed with given size.
plt.figure(figsize=(15, 5))

# To show original image with plt.
original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Shows original image.
plt.subplot(2, 3, 1)
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')

# DFT result is found by using dft function that uses grayscale of image and is provided by OpenCV
# to be able to call the implemented magnitude spectrum function and
# in order words, to use its result in my implementation function as input parameter.
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dft_result = cv2.dft(np.float32(image_grayscale), flags=cv2.DFT_COMPLEX_OUTPUT)

# The implemented magnitude spectrum function is used to display magnitude spectrum of image.
# Magnitude spectrum result is achieved using my magnitude spectrum function implementation.
magnitude_result = magnitude_spectrum(dft_result)
# Shows magnitude spectrum of the image using implemented magnitude spectrum function.
plt.subplot(2, 3, 2)
plt.imshow(magnitude_result, cmap='gray')
plt.title('Magnitude Spectrum - My Implementation')
plt.axis('off')

# Image that is read used as the parameter of normalize image function.
# Normalized image below is achieved using my normalize image implementation.
normalized_image = normalize_image(image)
#Shows normalized version of the image using implemented normalize function.
plt.subplot(2, 3, 3)
plt.imshow(normalized_image)
plt.title('Normalized Image - My Implementation')
plt.axis('off')

# Magnitude spectrum of the image using original openCV function is displayed
# to compare with my implementation magnitude spectrum function.
# In original OpenCV magnitude function it requires np fft shift function as input.
# Then, scales it with log-transform.
dft_shift_CV= np.fft.fftshift(dft_result)
magnitude_spectrum_openCV = 20 * np.log(cv2.magnitude(dft_shift_CV[:, :, 0], dft_shift_CV[:, :, 1]))
plt.subplot(2, 3, 5)
plt.imshow(magnitude_spectrum_openCV, cmap='gray')
plt.title('Magnitude Spectrum - OpenCV Function')
plt.axis('off')

# Normalized image by using original openCV function is displayed
# to compare with my implementation normalize function.
normalized_image_openCV = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.subplot(2, 3, 6)
plt.imshow(normalized_image_openCV)
plt.title('Normalized Image - OpenCV Function')
plt.axis('off')

plt.tight_layout()
plt.show()
