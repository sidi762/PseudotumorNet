import SimpleITK as sitk
import numpy as np
#Todo: reimplement using nibabel

#imagePath = './sample_image/MRI_extracted/jiaozhiliu/hechengxi_0019066268-aft/AX_T1+C_0007_brain'
#image = sitk.ReadImage(imagePath + '.nii')
#maskPath = './sample_image/segmented/hechengxi'
#mask = sitk.ReadImage(maskPath + '.nii')

def augmentationWithFactorArray(imageArray, maskArray, k):
    maskArray = np.array(maskArray, dtype=np.float32)
    imageArray = np.array(imageArray, dtype=np.float32)
    #print(maskArray[8][363][181])
    maskArray /= 2;
    #print(maskArray[8][363][181])
    augmentationFactor = 0.5 # k=0.5
    resultArray = imageArray + imageArray * maskArray * augmentationFactor
    #print(resultArray[8][363][181])
    return resultArray

def augmentationWithFactorArrayTwoChannel(imageArray, maskArray, k):
    maskArray = np.array(maskArray, dtype=np.float32)
    imageArray = np.array(imageArray, dtype=np.float32)
    print(imageArray.shape)
    #print(maskArray[8][363][181])
    maskArray /= 2;
    #print(maskArray[8][363][181])
    augmentationFactor = 0.5 # k=0.5
    imageArrayCH1 = imageArray[0,0,:,:,:]
    imageArrayCH2 = imageArray[0,1,:,:,:]
    resultArrayCH1 = imageArrayCH1 + imageArrayCH1 * maskArray * augmentationFactor
    resultArrayCH2 = imageArrayCH2 + imageArrayCH2 * maskArray * augmentationFactor
    resultArray = np.stack((resultArrayCH1, resultArrayCH2), axis=3)
    #print(resultArray[8][363][181])
    return resultArray


def augmentationWithFactor(image, mask, k):
    """
    Combine the segmentated lesion area with the original MR image
    to change the pixels of the lesion area by a multiple K
    M_aug = M + M * n * k

    image: sitk original image (sitk.ReadImage())
    mask: sitk mask image (sitk.ReadImage())
    k: factor, M_aug = M + M * n * k
    """

    print('image size:', image.GetSize())
    print('mask size:', mask.GetSize())

    #print(image.GetPixel((181,363,8)))
    #print(mask.GetPixel((181,363,8)))

    #print("image and mask info")
    #print(image)
    #print(mask)

    imageArray = sitk.GetArrayFromImage(image)# get numpy array from sitk image, float32
    maskArray = sitk.GetArrayFromImage(mask).astype(np.float32)

    resultArray = augmentationWithFactorArray(imageArray, maskArray, k)
    result = sitk.GetImageFromArray(resultArray)
    #Sync Metadata
    result.SetSpacing(image.GetSpacing())
    result.SetOrigin(image.GetOrigin())

    #Save
    #sitk.WriteImage(result, imagePath + '_augmented.nii.gz')
    #print('Augmented Image written to ' + imagePath + '_augmented.nii.gz' )
    return result
