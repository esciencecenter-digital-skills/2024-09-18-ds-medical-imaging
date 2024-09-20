# Medical Image Processing: the pilot version

:::info
This course is in pilot version. Expect a few imperfections and/or surprises. 

If you have never used a colllaborative document, it will help you to know that this document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.


:::


# Medical Image Processing Collaborative Document

09-18-2024 Medical Image Processing (day 1 of 1).

Welcome to The Workshop Collaborative Document.



----------------------------------------------------------------------------

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).


##  :dizzy:Advanced materials

There is a variety of skill levels in any course. Some more advanced learners will complete all the material, and can use our advanced materials. https://tinyurl.com/imaging-advanced


## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop websites

[lesson link](<https://esciencecenter-digital-skills.github.io/medical-image-processing/>)

[website link](<https://github.com/esciencecenter-digital-skills/medical-image-processing>)

🛠 Setup

[link](<https://esciencecenter-digital-skills.github.io/medical-image-processing/#installing-the-python-environment>)

Downloading files: Some data files can be obtained through the accompanying repository
[link](<https://github.com/esciencecenter-digital-skills/medical-image-processing-materials>)

## 👩‍🏫👩‍💻🎓 Instructors

Giulia Crocioni, Candace Makeda Moore

## 🧑‍🙋 Helpers

Ou Ku  

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ Organization/ pronouns

Removed for archiving.


## 💻👩‍💼🔬👨‍🔬🧑‍🔬🚀🔧 Ice-breaker
Turn the room into a map of the Netherlands  🧭 !

Stand in the city you came here from this morning (North is front of room)




## 🗓️ Agenda
| Time | Topic            |
| -------------:|:-------------------------- |
|
|          9:30 | Welcome and Intro  (Makeda)   |
|09:45 |	Course and Center Introduction|
|10:00|Introduction to Medical Imaging|
|10:30 |	MRI I|
|11:00 	|Coffee Break|
|11:15 	|MRI II|
|12:30| 	Lunch|
|13:30 |	Registration and Segmentation with SITK|
|15:30 	|Coffee Break|
|15:40| 	Preparing Images for ML|
|16:00 |	Anonymization|
|16:30 	|Generative AI|
|16:45 |	Wrap-up|
|17:00 	|Drinks |

 

## 🎓🏢 Evaluation logistics
* At the end of the day you should write evaluations into the colaborative document. You will get an email with an evluation/feedback form as well. The more information you give us the better you can help us make the course.


## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .


## 🎓🔧Evaluations

 
Evaluation | specific part or all
 Removed for archiving
 

## 🔧 Exercises

#### Exercise 1

Given that ultrasounds images are operator-dependent, often with embedded patient data, and the settings and patients’ positions can vary widely what can we do to optimize our research involving ultrasounds in terms of these limitations?

Answers removed for archiving.

#### Exercise 2

Check out the attributes of the array, what are the number of dimensions of our image?

Answers removed for archiving.

#### Exercise 3

What is the dimension and shape of our image (`t2_data`)?

Answers removed for archiving.

#### Exercise 4

Load into a variable the 20th slice from the y-axis,
then load into another variable the 4th slice from the x-axis

Answers removed for archiving.

### Preparing Images for Machine Learning

#### Exercise 5

Imagine you got the above images and many more because you have been assigned to make an algorithm for cardiomegaly detection. The goal is to notify patients if their hospital-acquired X-rays, taken for any reason, show signs of cardiomegaly. This is an example of using ML for opportunistic screening.

You are provided with two datasets:

- A dataset of healthy (no cardiomegaly) patients from an outpatient clinic in a very poor area, staffed by first-year radiography students. These patients were being screened for tuberculosis (TB).
- A dataset of chest X-rays of cardiomegaly patients from an extremely prestigious tertiary inpatient hospital.

#### Exercise 6

Use `skimage.transform.rotate` to create two realistic augmented images from the given ‘normal’ image stored in the variables.

Then, in a single block of code, apply what you perceive as the most critical preprocessing operations to prepare these images for classic supervised ML.

Hint: Carefully examine the shape of the cardiomegaly image. Consider the impact of harsh lines on ML performance.

##### Participants Answers

Removed for archiving

##### Our solution
Removed for archiving

### Anonymization

#### Exercise 7

How can you access and print additional patient identifying data? Hint: Refer to the documentation and compare with what we have already printed.

##### Participants Answers
Removed for archiving
##### Our solution
Removed for archiving

## 🧠 Collaborative Notes

### MRI

Activate the environment
```bash
conda activate medical_image_proc
```

Get to the help page of `dcm2niix`
```bash
dcm2niix -help
```

start a jupyter lab session
```bash
jupyter lab
```

Create a new Jupyter Notebook

Importing the following packages:

```python
# import libraries
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
```

Load a T2 image
```python
t2_img=nib.load("data/mri/OBJECT_phantom_T2W_TSE_Cor_14_1.nii")
```

Get the hearder and print
```python
t2_data_header = t2_img.header
print(t2_data_header)
```

```python
print(t2_data_header.values()[7])
```

```python
# print out the `pixdim` field
print(t2_data_header['pixdim'])
```

Access image

```python
# This prints out an array
t2_data = t2_img.get_fdata()
print(t2_data)
```

```python
# Inspect the data format
print(type(t2_data))
```

```python=
# Get the image proxy
t2_img.dataobj
```

You get the menu of attributes by pressing "Tab" after the dot "."
```python
#t2_data.(Tab here)
```

Get the number of dims and shape of the image
```python=
print(t2_data.ndim)
print(t2_data.shape)
```

```python=
# Get the dtype of the numpy array
# Shoul give float64
print(t2_data.dtype)
```

```python=
# Get min max values
print(np.min(t2_data))
print(np.max(t2_data))
```

```python=
# Get values at one location by 3d indexing 
print(t2_data[9,19,2])
```

```python=
# 10th slice in z axis
z_slice = t2_data[:,:,9]
```

```python=
y_slice = t2_data[:, 19, :] # 20th slice in y
x_slice = t2_data[3, :, :] # 4th slice in x
```

Plot wit Jupyter magic command

```python=
%matplotlib inline
slices = [z_slice, x_slice, y_slice]
fig, axes = plt.subplots(1, len(slices))
for i, slice in enumerate(slices):
    axes[i].imshow(slice.T, cmap="gray", origin="lower")
```


```python=
# Get the affine matrix
# It indicates the position of the image array data in a reference space
t2_affine = t2_img.affine
print(t2_affine)
```

### Registration and Segmentation with SITK


We will use library called sitk. 
Registration and segmentation operations are both very common in medical image analysis. We see images that need a tranformation to be superimposed: that is a regiatration. 
(Image of segmented brain). We will use sitk, a simplified interface to ITK. 

Fundamental concepts: 
Images in sitk occupies a physical space. COncpets such as origin, spacing relate the pixels/voxels to physical space.

(Image of grid with pixel coordinate) Sitk has spacing, therefore you may have cases where certain packages are read as image array, and you could get a distorted images. SITK will give an undistorted image because it is "aware" of spacing on different axes. 

Concept: transforms/ transformations
 example: translation, rotation,m affine
 
 Bounded domain transformations
 Appled only to an read or identity
 
 Composite: mixes both bounded and 'global' ones

Concept: Resampling

Resampling grid : we need to make everything match by downsampling or oversampling (operations implemented in library). We can use different interpolators in sitk

Registration:
Perfectly overlapping images requires registration

Segmentation:
Extracts a certain area of the brain. Various techniques available.

Coding:

```python=

%matplotlib inline

import matplotlib.pyplot as plt
import SimpleITK as sitk


# read image
img_volume = sitk.ReadImage("data/sitk/A1_grayT1.nrrd")

print(type(img_volume))
print(img_volume.GetOrigin())
print(img_volume.GetSpacing())
print(img_volume.GetDirection())

# print more stuff

print(img_volume.GetSize())
print(img_volume.GetWidth())
print(img_volume.GetHeight())
print(img_volume.GetDepth())
print(img_volume.GetNumberOfComponentsPerPixel())

```

Note indexing is zero based like in numpy. SITK has a built in methods for showing/displaying. Let's examine


```python
sitk.Show?
```

This will  work with an external display. We will take another method

```python
import numpy as np

multi_channel_3D_image = sitk.Image([2,4,8], sitk.sitkVectorFloat32, 5)

x = 1 
y = 3
z = 7

multi_channel_3D_image[x,y,z] = np.random.random(multi_channel_3D_image.GetNumberOfComponentsPerPixel())

nda = sitk.GetArrayFromImage(multi_channel_3D_image)

print("Image size:", str(multi_channel_3D_image.GetSize()))
print("Numpy array size:",  str(nda.shape))
```

Let's transform our volume into an array


```python
npa = sitk.GetArrayFromImage(img_volume)
npa_zslice = sitk.GetArrayFromImage(img_volume[z,:,:])
z = int(img_volume.GetDepth()/2)
npa_slice = sitk.GetArrayFromImage(img_volume[z,:,:])

fig = plt.figure(figsize=(10,3))


fig.add_subplot(1,3,1)
plt.imshow(npa_zslice)
plt.title("default colormap", fontsize=10)
plt.axis=('off')

fig.add_subplot(1,3,2)
plt.imshow(npa_zslice, cmap=plt.cm.Greys_r)
plt.title("grey colormap", fontsize=10)
plt.axis=('off')

fig.add_subplot(1,3,3)
plt.imshow(npa_zslice, cmap=plt.cm.Greys_r, vmin=npa.min(), vmax=npa.max())
plt.title("grey colormap based on min and max calues", fontsize=10)
plt.axis=('off')
```
We can also reverse the process


```python
img_zslice = sitk.GetImageFromArray(npa_zslice)
print(type((img_zslice)))
```

```python
img_xslices = [img_volume[x,:,:] for x in range(50,200,30)]
img_yslices = [img_volume[:,y,:] for y in range(50,200,30)]
img_zslices = [img_volume[:,:,z] for z in range(50,200,30)]


tile_x = sitk.Tile(img_xslices, [1,0])
tile_y = sitk.Tile(img_yslices, [1,0])
tile_z = sitk.Tile(img_zslices, [1,0])

nda_xslices = sitk.GetArrayViewFromImage(tile_x)
nda_yslices = sitk.GetArrayViewFromImage(tile_y)
nda_zslices = sitk.GetArrayViewFromImage(tile_z)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(nda_xslices, cmap=plt.cm.Greys_r)
ax2.imshow(nda_yslices, cmap=plt.cm.Greys_r)
ax3.imshow(nda_zslices, cmap=plt.cm.Greys_r)

```

Now let's see examples of operations that exploit numpy capabilities

```python
n_slice = 150
plt.imshow(sitk.GetArrayViewFromImage(img_volume[:,:,n_slice]), cmap=plt.cm.Greys_r)
```
Let's do operations on numpy array
```python

plt.imshow(sitk.GetArrayViewFromImage(img_volume[:,:100, n_slice]), cmap=plt.cm.Greys_r)
```

Let's so a few more operations

```python

plt.imshow(sitk.GetArrayViewFromImage(img_volume[:,::3, n_slice]), cmap=plt.cm.Greys_r)
```
The above subsamples image so it is squeezed along y-axis.
Now let's see how we could flip the image:
```python

plt.imshow(sitk.GetArrayViewFromImage(img_volume[:,::-1, n_slice]), cmap=plt.cm.Greys_r)
```

Now a proper sampling example:

```python

img_volume = sitk.ReadImage("data/sitk/training_001_mr_T1.mha")
print(img_volume.GetSize())
print(img_volume.GetSpacing())

img_xslices = [img_volume[x,:,:] for x in range(50,200,30)]
img_yslices = [img_volume[:,y,:] for y in range(50,200,30)]
img_zslices = [img_volume[:,:,z] for z in range(1,25,3)]



tile_x = sitk.Tile(img_xslices, [1,0])
tile_y = sitk.Tile(img_yslices, [1,0])
tile_z = sitk.Tile(img_zslices, [1,0])

nda_xslices = sitk.GetArrayViewFromImage(tile_x)
nda_yslices = sitk.GetArrayViewFromImage(tile_y)
nda_zslices = sitk.GetArrayViewFromImage(tile_z)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(nda_xslices, cmap=plt.cm.Greys_r)
ax2.imshow(nda_yslices, cmap=plt.cm.Greys_r)

def resample_img(volume, out_spacing=[1.25,1.25,1.25]):
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    
    out_size= [
        int(np.round(original_size[0]) * (original_spacing[0]/out_spacing[0])),
        int(np.round(original_size[1]) * (original_spacing[0]/out_spacing[0])),
        int(np.round(original_size[2]) * (original_spacing[0]/out_spacing[0])),
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(volume.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
        
    return resample.Execute(volume)

resampled_sitk_img = resample_img(img_volume)
print(resampled_sitk_img.GetSize())
print(resampled_sitk_img.GetSpacing())
```

Let's start with registration: 

Here are the basic elements: We will have a transformation element. Then you need to define a metric to define if the image is well registered or not. We then need to to minimize or maximize your metrix. The interpolation to resample for teh registration. 

```python

from ipywidgets  import interact, fixed
from IPython.display import clear_output
import os

OUTPUT_DIR = "data/sitk/"
fixed_image = sitk.ReadImage(f"{OUTPUT_DIR}training_001_ct.mha", sitk.sitkFloat32)

moving_image = sitk.ReadImage(f"{OUTPUT_DIR}training_001_mr_T1.mha", sitk.sitkFloat32)

def display_image(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    plt.subplots(1,2,figsize=(10,8))
    
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_img_z,:,:], cmap=plt.cm.Greys_r)
    plt.title('fixed/target image')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_img_z,:,:], cmap=plt.cm.Greys_r)
    plt.title('moving/source image')
    plt.axis('off')
    
    plt.imshow()
interact(
    display_image,
    fixed_image_z =(0,fixed_imag.GetSize()[2]-1), 
    moving_image_z = (0,moving_image.GetSize()[2]-1),
    fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)),
    moving_npa = fixed(sitk.GetArrayViewFromImage(moving_image)))

)
    
# rest of code to be added
```

Now let's register.

```python

initial_transforms = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)
moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    initial_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID())

def display_images_with_alpha(image, alpha, fixed, moving):
    img = (1 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z]
    plt.imshow(sitk.GetArrayFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis = ('off')
    plt.show()
    
interact(
    display_images_with_alpha,
    image_z = (0, fixed_image.GetSize()[2]-1),
    alpha = (0,1.0,0.05),
    fixed =fixed(fixed_image),
    moving = fixed(moving_resampled)
)
    # superimpose
    
```
So now let's define a metric, an interpolatpor and optimizer

```python
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
    

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()


registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()


registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.            
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace=False)


# Connect all of the observers so that we can perform plotting during registration.
registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                               sitk.Cast(moving_image, s
                                                         
                                                         itk.sitkFloat32))
```

Let's print out what we get

```python
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
```

Let's inspect visually

```python
moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID())

interact(
    display_images_with_alpha,
    image_z = (0,fixed_image.GetSize()[2]-1),
    alpha = (0.0,1.0,0.05),
    fixed = fixed(fixed_image),
    moving = fixed(moving_resampled))

```

Let's examine more closely:


```python
print(f"Origin for fixed image: {fixed_image.GetOrigin()}, shifted moving image: {moving_resampled.GetOrigin()}")
print(f"Spacing for fixed image: {fixed_image.GetSpacing()}, shifted moving image: {moving_resampled.GetSpacing()}")
print(f"Size for fixed image: {fixed_image.GetSize()}, shifted moving image: {moving_resampled.GetSize()}")

OUTPUT
```
We can write the image
```python
sitk.WriteImage(moving_resampled, os.path.join(OUTPUT_DIR, 'RIRE_training_001_mr_T1_resampled.mha'))
sitk.WriteTransform(final_transform, os.path.join(OUTPUT_DIR, 'RIRE_training_001_CT_2_mr_T1.tfm'))
```

Segmentation: 

```python
%matplotlib inline
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import SimpleITK as sitk

img_T1 = sitk.ReadImage("data/sitk/A1_grayT1.nrrd")
# To visualize the labels image in RGB with needs a image with 0-255 range
img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# a volume image
def display_images(image_z, npa, title):
    plt.imshow(npa[image_z,:,:], cmap=plt.cm.Greys_r)
    plt.title(title)
    plt.axis('off')
    plt.show()

interact(
    display_images,
    image_z = (0,img_T1.GetSize()[2]-1),
    npa = fixed(sitk.GetArrayViewFromImage(img_T1)),
    title = fixed('Z slices'))
```

### Preparing Images for Machine Learning



#### Basic steps

Let’s go throught some examples. First, we import the libraries we need:

```python
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage import transform
from skimage import io
from skimage.transform import rotate
from skimage import transform as tf
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import resize
```

Then, we import our example images and examine them.

```python
image_cxr1 = io.imread('data/ml/rotatechest.png') # a relatively normal chest X-ray (CXR)
image_cxr_cmegaly = io.imread('data/ml/cardiomegaly_cc0.png') # cardiomegaly CXR
image_cxr2 = io.imread('data/ml/other_op.png')
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 1
columns = 3

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(image_cxr1)
plt.axis('off')
plt.title("Normal 1")
  
# add a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
# showing image
plt.imshow(image_cxr_cmegaly)
plt.axis('off')
plt.title("Cardiomegaly")

# add a subplot at the 3nd position
fig.add_subplot(rows, columns, 3)
# showing image
plt.imshow(image_cxr2)
plt.axis('off')
plt.title("Normal 2")
```

### Anonymizing Medical Images

Now, let’s see how to open a DICOM file and work with it using Pydicom.

First, let’s import Pydicom and read in a CT scan:

```python
import pydicom
from pydicom import dcmread
fpath = "data/anonym/our_sample_dicom.dcm"
ds = dcmread(fpath)
print(ds)
```

```output
Dataset.file_meta -------------------------------
(0002, 0000) File Meta Information Group Length  UL: 218
(0002, 0001) File Meta Information Version       OB: b'\x00\x01'
(0002, 0002) Media Storage SOP Class UID         UI: CT Image Storage
(0002, 0003) Media Storage SOP Instance UID      UI: 1.3.46.670589.33.1.63849049636503447100001.4758671761353145811
(0002, 0010) Transfer Syntax UID                 UI: JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])
(0002, 0012) Implementation Class UID            UI: 1.2.840.113845.1.1
(0002, 0013) Implementation Version Name         SH: 'Syn7,3,0,258'
(0002, 0016) Source Application Entity Title     AE: 'SynapseDicomSCP'
-------------------------------------------------
(0008, 0005) Specific Character Set              CS: 'ISO_IR 100'
(0008, 0008) Image Type                          CS: ['DERIVED', 'SECONDARY', 'MPR']
[...]
```

We can modify elements of our DICOM metadata:

```python
elem = ds[0x0010, 0x0010]
print(elem.value)
```

```output
OurBeloved^Colleague
```

```python
elem.value = 'Citizen^Almoni'
print(elem)
```

```output
(0010, 0010) Patient's Name                      PN: 'Citizen^Almoni'
```

You can also just set an element to empty by using `None`:

```python
ds.PatientName = None
print(elem)
```

```output
(0010, 0010) Patient's Name                      PN: None
```

## 📚 Resources

Our didactic website has a glossary. This is available from the [lesson link](<https://esciencecenter-digital-skills.github.io/medical-image-processing/>). If you see a term which confuses you, you can help us by asking about it. This is a pilot, and we hope to learn from you about how to help future students learn. Please add words you want in the glossary below.

- [nipreps](https://www.nipreps.org/)
    
    
[Introduction to Working with MRI Data in Python](https://carpentries-incubator.github.io/SDC-BIDS-IntroMRI/)

[Introduction to dMRI](https://carpentries-incubator.github.io/SDC-BIDS-dMRI/)

[Functional Neuroimaging Analysis in Python](https://carpentries-incubator.github.io/SDC-BIDS-fMRI/)

Pipeline resources: 

[Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki)
[FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/)

Glossary suggestions:

Sinogram


## 🧠📚 Final tips and tops
Removed for archiving
