# -*- coding: utf-8 -*-

##################import Libraries##################################
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob

from datetime import timedelta

"""------------------------------"""
import time
import os
import sys

class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(sys.stdout)  #  将输出记录到log
sys.stderr = Logger(sys.stderr)  # 将错误信息记录到log

"""--------------------------------"""

#########################Import Libaries#############################
start_time = time.time()

# =============================================================================
# Function Definitions
# =============================================================================

# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).

#############################Functions Done####################################

# =============================================================================
# Input
# =============================================================================
# fixed_image = sitk.ReadImage('result/BSDS2XJYF/combined/20210226_M082Y_WANG_FURUN_X_t1flair.nii', sitk.sitkInt16)
# moving_image = sitk.ReadImage('result/BSDS2XJYF/combined/20210226_M082Y_WANG_FURUN_X_t2flair.nii', sitk.sitkInt16)

# Transformation_imageName='Fix__moving_'+filename2[:5]
### Shoe the Image
# interact(display_images, fixed_image_z = (0, fixed_image.GetSize()[2] - 1),
#          moving_image_z = (0, moving_image.GetSize()[2] - 1), fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)),
#          moving_npa = fixed(sitk.GetArrayViewFromImage(moving_image)));
#################################################Input Done ############################################

def read(path):
    return sitk.ReadImage(path, sitk.sitkInt16)

paths = glob("../../Dataset/RESECT/resize/test/*")

for path in paths:
    t1path = glob(path+"/*T1.nii")[0]
    flairpath = glob(path + "/*FLAIR.nii")[0]
    nt1path = t1path.replace("resize","resize_affine")
    nflairpath = flairpath.replace("resize", "resize_affine")
    t1dir = os.path.split(nt1path)[0]
    if not os.path.exists(t1dir):
        os.mkdir(t1dir)
    moving_image = read(t1path)
    fixed_image = read(flairpath)
    print(fixed_image.GetSize())
    print(t1dir)

    ##Name the Image
    # =============================================================================
    # Registartion Start
    # =============================================================================
    # registration Method.
    start_time = time.time()
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    #############################Initializing Initial Transformation##################################
    grid_physical_spacing = [150.0, 150.0, 150.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, transformDomainMeshSize = mesh_size, order=2)
    
    registration_method.SetInitialTransform(initial_transform)

    #######################Matrix###################################################3
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.3)

    ##################Multi-resolution framework############3
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    ##############Interpolation#################################
    registration_method.SetInterpolator(sitk.sitkLinear)

    ##################Optimizer############################
    # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate = 1.5, numberOfIterations = 6,
                                                      convergenceMinimumValue = 1e-4,
                                                      convergenceWindowSize = 5)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    #######################################Print Comment#############################################
    # # Connect all of the observers so that we can perform plotting during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    #################Transformation###################################################################
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    # =============================================================================
    # post processing Analysis
    # =============================================================================
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    # Visualize Expected Results
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.2,
                                     moving_image.GetPixelID())
    # moving_resampled2 = sitk.Resample(fixed_image, moving_image, final_transform, sitk.sitkLinear, 0.2,
    #                                   moving_image.GetPixelID())
    # moving_resampled3 = sitk.Resample(fixed_image, moving_image, final_transform, sitk.sitkLinear, 0.2,
    #                                   fixed_image.GetPixelID())

    # interact(display_images_with_alpha, image_z = (0, fixed_image.GetSize()[2]), alpha=(0.0, 1.0, 0.05),
    #          fixed=fixed(fixed_image), moving = fixed(moving_resampled))

    ################Saving Transformed images###################################33
    sitk.WriteImage(moving_resampled, nt1path)
    sitk.WriteImage(fixed_image, nflairpath)
    # sitk.WriteImage(moving_resampled2,Registered_imageName+'_two' +'.nii.gz')
    # sitk.WriteImage(moving_resampled3,Registered_imageName +'_three'+'.nii.gz')

    elapsed_time_secs = time.time() - start_time
 
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

    print(msg)
 
