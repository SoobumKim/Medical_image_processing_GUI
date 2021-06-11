# Medical_image_processing_GUI

## MR DICOM GUI for image processing

<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121631197-960b1000-cab9-11eb-8ec1-de82e18f1df8.png width="700" height="400"></>
  
You can process MR Dicom images easily using several methods in this GUI.
 
### Menubar_File
  > File
>	> Open : Dicom series open
  
>	>	Save : Save the processed images
  
### Menubar_Filter
  
  > Filter
> > Mean filter
> > > Get size 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121640389-d2923800-cac8-11eb-9b12-201bf1bb9377.png width="350" height="180"></>  

> > Max filter
> > > Get size
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121640454-e8076200-cac8-11eb-82ea-ae6adceb8fa9.png width="350" height="180"></>  
  
> > Median filter 
> > > Get size
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121640488-f5bce780-cac8-11eb-9dc1-097d19f239c1.png width="350" height="180"></>  

> > Sobel filter 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121642046-f22a6000-caca-11eb-8968-aaeb67eea485.png width="350" height="180"></>  
  
> > Prewitt filter
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121640585-1422e300-cac9-11eb-90b9-83c5ef46c0bf.png width="350" height="180"></>

> > Canny filter 
> > > Get sigma
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121640645-256bef80-cac9-11eb-9faf-67966d761adc.png width="350" height="180"></>
  
> > Log filter
> > > Get sigma 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121640688-33ba0b80-cac9-11eb-8f9b-f84988293bec.png width="350" height="180"></>  

### Menubar_Image Enhancement
  > Image Enhancement
> > Image Inverse 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641047-a6c38200-cac9-11eb-84af-aa74bbd27e33.png width="350" height="180"></>  
  
> > Power Law Transformation 
> > Get Gamma 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641092-b642cb00-cac9-11eb-8051-58e7deeed68b.png width="350" height="180"></>  

> > Log Transformation 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641127-c2c72380-cac9-11eb-9381-b451373d8ce1.png width="350" height="180"></>  
  
> > Histogram Equalization 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641172-cfe41280-cac9-11eb-8f74-32a2f1c9fc69.png width="350" height="180"></>  

> > Contrast Stretching 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641211-dc686b00-cac9-11eb-944c-3fb203767112.png width="350" height="180"></>  

### Menubar_Fourier Transform
  > Fourier Transform
> > Low pass filter
> > > Ideal low pass filter
> > > > Get Cut-off
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641266-ef7b3b00-cac9-11eb-9f79-1a74993af220.png width="350" height="180"></>  

> > > Butterworth low pass filter
> > > > Get Cut-off
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641313-fbff9380-cac9-11eb-8c15-5b1a1ffb66e0.png width="350" height="180"></>  
  
> > > Gaussian low pass filter
> > > > Get Cut-off
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641438-25202400-caca-11eb-9608-277617628c43.png width="350" height="180"></>  

> > High pass filter
> > > Ideal high pass filter
> > > > Get Cut-off
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641465-2f422280-caca-11eb-85ce-3124048d1e70.png width="350" height="180"></>  
  
> > > Butterworth high pass filter
> > > > Get Cut-off
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641531-3ff29880-caca-11eb-8205-a098175cfa08.png width="350" height="180"></>  

> > > Gaussian high pass filter
> > > > Get Cut-off
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641563-4b45c400-caca-11eb-982f-47837dc74580.png width="350" height="180"></>  
  
> > Band pass filter
> > > Get minimum
> > > Get maximum
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641598-5698ef80-caca-11eb-99f0-e78863381f47.png width="350" height="180"></>  

### Menubar_Segmentation
  > Otsu Method
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641633-6284b180-caca-11eb-8f63-e2f60aedd5c2.png width="350" height="180"></>  
  
  > Renyi Entropy
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641670-716b6400-caca-11eb-891a-bbf2412a231d.png width="350" height="180"></>  

  > Watershed Segmentation
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641715-7fb98000-caca-11eb-9e6e-4fa0943fa059.png width="350" height="180"></>  
  
### Menubar_Morphological
  > Dilation
> > iterations
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641767-91028c80-caca-11eb-8024-527e3e274575.png width="350" height="180"></>  

  > Erosion
> > iterations  
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641798-9c55b800-caca-11eb-83c4-0d0f3f08d768.png width="350" height="180"></>  
  
  > Opening
> > iterations  
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641832-a7a8e380-caca-11eb-9f6b-09299f81cb03.png width="350" height="180"></>  

  > Closing
> > iterations 
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641871-b5f6ff80-caca-11eb-90a5-fc0fc8fa92e0.png width="350" height="180"></>  
  
  > Hit or Miss
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641905-c018fe00-caca-11eb-883e-1b61e6999645.png width="350" height="180"></>  

  > Skeletonize
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121641937-cb6c2980-caca-11eb-8c13-8fb6670f63b5.png width="350" height="180"></>  
  
### Button
  > Reset : Reset from your dicom to orignal dicom
  
  > Up : Change dicom of Before and After to above dicom (-after apply any process method, display anything in 'After' block)
  
  > Down : Change dicom of Before and After to below dicom
  
# DICOM DATA
Reference - http://www.pcir.org/researchers/downloads_available.html
