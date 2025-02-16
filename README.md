# Image Processing: Magnitude and Normalize Functions as Alternatives of Ready-made OpenCv Functions
This project focuses on the two functions which are the magnitude and normalize. The magnitude function provides to find the magnitude spectrum of an image by taking the input as the DFT result of an image which is a 2D complex matrix. It calculated the magnitude of each complex number. The normalize function allows to normalization of the given image by setting the mean value of the image to 0 and the standard deviation of the image to 1. These are the alternative forms of OpenCV functions. The mathematical perfectives, approaches, complexities, and results of these functions are examined.

# 1. Mathematical Properties
# 1.1 Magnitude Function: 
This function takes a DFT result of an image as an input. DFT result is the Discrete Fourier Transform of the image. When DFT is applied to an image, it will result in a complex 2D matrix. Then magnitude function provides to take the magnitude of these complex numbers and gives the output. It converts the DFT results to shifted DFT results. Then, it finds the magnitude of each complex number to find the magnitude result. But this result cannot be directly displayed. That’s why after that the ‘magnitude_spectrum’ function takes the log transform of the magnitude result to get a more homogeneous magnitude spectrum visualization. 
The mathematical calculation of the magnitude of each pixel as a complex number can be seen below.

For a 1D complex number:

![image](https://github.com/user-attachments/assets/89ad62e8-132f-4cc6-87d5-2deab48a88f5)

(z = complex number, a = real part, b = imaginary part)


For image 2D complex numbers are used as:

![image](https://github.com/user-attachments/assets/a3e51748-32f4-4f91-b080-db0e4c40737d)


(F(x,y) = complex number, A(x,y) = real part, B(x,y) = imaginary part)

Overall, the magnitude function computes the magnitude of each complex number in the shifted DFT result of the image. It returns another 2D matrix that has the same size as the DFT matrix and gives the magnitude of the corresponding image. Then, the log transform of magnitude spectrum values is used to get the data to visualize the magnitude spectrum.

# 1.2 Normalize Function:

The normalize function provides to scale the image data to a standard range. It sets the mean of the image to 0 and the standard deviation of the image to 1. This ensures that the image data is at the same scale and therefore this function is often used to standardize the data before some applications.

This operation includes working on each pixel value of an image one by one. For each pixel, it should be estimated by using pixel value, mean, and standard deviation. The approach will be changed depending on whether the image is colorful or not. The new value of the current pixel value should be calculated by subtracting the original pixel value and mean, and then dividing it by the standard deviation. The achieved x’ value will be the normalized x value.

![image](https://github.com/user-attachments/assets/3ba30a97-1c49-4238-a55c-4504b683c396)


(x’ = normalized pixel value, x = current pixel value, μ = mean, σ = standard deviation)

So, to normalize an image the steps should be realized are:
	1.Calculate μ (mean) and σ (standard deviation) of original image.
	2.Subtract the mean value from the pixel value one by one.
	3.Dive each pixel value by the standard deviation one by one.
	4.The mean of the image is set to 0 and the standard deviation of the image is set to 1.

# 2. Approach
# 2.1 Magnitude Function:

This function estimates the magnitude spectrum of an image by using the DFT result. DFT result that is a 2D complex matrix as mentioned is the input of the function. It gives the magnitude spectrum as the output of the image using the DFT result.

Detailed steps:
1.	Firstly, the magnitude spectrum function takes the DFT result of an image as an input parameter. 
2.	Then, it changes the DFT result for each pixel which is implemented in the ‘magnitude_spectrum’ function to get the shifted version of the taken DFT result. This makes the data more understandable to get the magnitude spectrum.
3.	Then, it scans whole rows and columns one by one for each pixel. Each cell has a complex number.
4.	The magnitude of each complex number is estimated by taking the square root of the sum of the squares of the real and imaginary parts.
5.	Calculated values are added to a matrix and the other implemented part of the function which provides to find the log transform of calculated magnitudes is executed to get a more homogenous magnitude spectrum visual. Otherwise, the calculated magnitude spectrum result cannot give a remarkable magnitude spectrum solution. 
6.	After all, the log-transformed version of the calculated magnitude spectrum list is returned and the magnitude spectrum visual is displayed using the plt library. 

# 2.2 Normalize Function:

This function normalizes each pixel one by one by altering the mean value to 0 and the standard deviation to 1. It takes the image as an input and gives the normalized image as the output. It makes some additional operations if the image is colorful. 

Detailed steps:
1.	Estimates the mean and standard deviation values of the original image. 

       If the image is grayscale,
  
       1.1	it executes the calculations for each pixel one by one.
     
       If the image is colorful,
  
       1.1 it executes the calculation for each channel value of each pixel one by one.

2.	The normalized value for each pixel is achieved in two steps.
   
       If the image is grayscale:
   
       2.1	Firstly, the pixel value is subtracted from the mean value of the original image.
  
       2.2	Then, the result of the subtraction is divided into the standard deviation of the original image.
   
       If the image is colorful:
  
       2.1 Firstly, each channel value of each pixel is subtracted from the mean value of the original image.
  
       2.2	Then, the result of the subtraction is divided into the standard deviation of the original image.
   
       2.3 After the normalized pixel values are calculated for each pixel, the values of channels are scaled by using min and max values for each pixel. Otherwise, it does not give the proper normalized result.
 
 3.  The scaled final normalized resulting image is returned. The normalized image is displayed using the plt library. 

# 3. Computational Efficiency
# 3.1 Magnitude Function:

Since the ‘magnitude_spectrum’ function will travel on a 2D complex matrix and make fft shift, the function created by me has the time complexity O(m*n*log(n)), and the function available in the OpenCV library runs at O(m*n) time complexity. Likewise, the space complexity for both implementations is O(m*n). But of course, the ready function works better and faster. Because they are optimized and work more functionally by using many ready-made libraries. Also, the time complexity is optimized in OpenCV. In the function I created myself, no ready-made library was used. The finding magnitude spectrum needs several mathematical processes to scale the calculated magnitude spectrum data otherwise, data cannot make any sense visually. That’s why it may work slower than the ready one in some calculation parts.

Possible Optimizations for Magnitude Function: 
1.	Unnecessary calculations may be avoided.
2.	The used data structure can be changed with a more useful one.
3.	If it is appropriate, parallel processing can be used.
4.	Another way can be found to visualize the findings about the magnitude spectrum rather than log transform usage.

# 3.2 Normalize Function:

The ‘normalize_image’ function must scan the whole image matrix to calculate the mean and standard deviation of the original image. Additionally, it must estimate normalized pixel values by scanning each one. So, it means that since it must check channels for each pixel, its time complexity is O(m*n*c). m n caused by matrix and c caused by channel number. Also, the space complexity is O(m*n) because its input and output matrixes have the same sizes. In addition, the ready OpenCV function will also have the same space complexity but O(m*n) time complexity. The practically ready function will be faster than implemented by myself because in my function there is not any external library to make calculations rapidly. 

Possible Optimizations for Normalize Function:
1.	Providing the mean and standard deviation estimations in one pass will affect the time complexity due to avoiding repeated calculations. 
2.	Optimization of loops to scan the image matrixes will provide to reduce the space and memory usage.

# 4. Result and Analysis

The result is ready the OpenCV functions are more rapid than my implementations because the external libraries are not used in my implementations, and they are optimized. Implemented functions may need to be optimized to reduce the complexities. Because they require some processes to make their result visually remarkable. However, both of my implemented functions give correct image processing results with different tries. 

# 4.1 Magnitude Function:

The ‘magnitude_spectrum’ function provides to find the magnitude spectrum of the image and takes the DFT result of the image. But when it uses the DFT results directly, it does not give the visual of magnitude spectrum properly. That’s why the shifted version of the DFT result is used to find the magnitude spectrum of the original image. In other words, the DFT result is scaled to use it for finding the magnitude spectrum. Then, the magnitude result is calculated for each pixel represented by complex numbers. After that, the logarithmic transform is applied to the found magnitude result. It means that the proper magnitude spectrum result can be found just by the usage of a shifted version of DFT and logarithmic transform of the calculated magnitude result. If the magnitude result is not scaled with log transform and found using shifted DFT, the visual of the magnitude spectrum will not be visible as requested. 

# 4.2 Normalize Function:

The ‘normalize_image’ function provides the normalized version of the image by setting the mean value of the image to zero and the standard deviation of the image to one. This is generally a pre-processing operation. The function takes the image as the input and gives the normalized version of the image as the output. It estimates the mean and standard deviation values of the image and then, normalizes each pixel one by one by using those findings. Each normalized pixel value is added to a list just like the input version of the image. These normalized pixel values constructed the last version of the normalized image together. 

However, there is a difference between the application of the function to a colorful image and a grayscale image. If the image is colorful it means that it has 3 channels ‘R,G,B’. That’s why these should be considered one by one. In the grayscale image, there is no third channel. It makes calculations on each pixel for mean and standard deviation. The same process is applied to the colorful image as well. However, it cannot directly calculate for pixels because each pixel has 3 values. That’s why it should consider them one by one. Additionally, the normalized image of a colorful image needs to be scaled by using channel, max, and min values in the normalized image. Otherwise, it does not give the correct version of the normalization even if the calculations are true because of its being colorful.
The original version of the images used in this project and the result of the functions implemented and OpenCV as the magnitude spectrum of each image and normalized version of each image can be seen compared below respectively. 

![image](https://github.com/user-attachments/assets/f39435bd-4f4e-4bdf-b8d8-29988495e982)
                                            Figure-1
                                      

![image](https://github.com/user-attachments/assets/81686a2d-7e93-48cb-a155-bb8bbc393ec1)
                                      Figure-2


![image](https://github.com/user-attachments/assets/b9eb7153-d8e0-4470-ad42-ab42fd7c184f)
                                      Figure-3


![image](https://github.com/user-attachments/assets/24b2befb-1e50-4ce6-b4fa-a712137d64db)
                                      Figure-4

# 5. Conclusion
In conclusion, ready functions are better in terms of performance and memory usage, making them suitable for most applications. But my implementations that are ‘magnitude_spectrum’ and ‘normalize_image’ functions provide correct outputs. In addition, the versions developed provide flexibility on the algorithms as they can be changed, which means they may be better specifically for some applications. For future research and applications, optimizing the functions I have developed and using more developed techniques can improve the performance of the functions in terms of speed and memory. Recognizing both the strengths and limitations of each approach is crucial to making decisions appropriate to different projects. With the improvements to be made, alternatively developed functions may even work better than ready OpenCV solutions and provide more specific solutions to problems.
