This method is replicated version of Space Net 7 winner solution. We have used their post processing techniques, but we evaluate / predict using our own model.

Before running the code, make sure you have following things:
|
|___ All dependencies are installed. In this computer we have conda environment named 'solaris'
|
|___ All the target images, from all AOIs and dates are placed in a single folder "path_to_root"/"Images". Rename all Images as follows: (AOI_name)_(frame_num).
|	where frame_num is used to sort the images. Make sure you define it carefully, prefferable is to use: 00,01,02,03,....,16, that is make sure it is an int
|	where all values have same number of digits and not like 0,1,2,....,16 etc.
|
|___ Model is downloaded on your machine. In this machine it is placed @ 
|	"D:\LUMS_RA\Python Codes\Segmentation\Models\Keras\trained_model\DeepLabV3+\DHA_Built_vs_Unbuilt_temporal2"
|
|___ Depending on your size/number of images, your hardware must have enough space to store intermediate results.

