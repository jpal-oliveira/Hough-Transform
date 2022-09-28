import numpy as np
import math 
import pyopencl as cl
import cv2 as cv
import time

#include <fstream>

def GetFaceFromVideo(img):
    height, width = img.shape[0], img.shape[1]
    roi = img.copy()
    face_detected = False
    haar = cv.CascadeClassifier("...\\haarcascade_frontalface_alt2.xml") #Path to the file haarcascade_frontalface_alt2.xml
    faces = haar.detectMultiScale(img, scaleFactor = 1.2, minSize = (20, 20), maxSize = (width // 2, height // 2))
    for (x,y,w,h) in faces:
        img = cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
        roi = img[y:y+h, x:x+w]
        face_detected = True
    return img, roi, face_detected

if __name__ =="__main__":

    f = open("../config.txt", "r") #Path to the file config.txt
    lines = f.readlines()
    diff=lines[0].strip("diff:")
    average=lines[1].strip("average:")
    raio_min=int(lines[2].strip("raio_min:"))
    raio_max=int(lines[3].strip("raio_max:"))
    #print(diff, average, raio_min, raio_max)
    f.close()
    
    vidCap = cv.VideoCapture(0)

    if (not vidCap.isOpened()):
        print("Video File Not Found")
        exit(-1)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    with open("../Hough_Transform.txt", 'r') as file: #Path to the file haarcascade_frontalface_alt2.xml
        data = file.read()
    prg = cl.Program(ctx,data).build()

    while(True):
        ret, vidFrame = vidCap.read()

        start_time = time.time()

        if (not ret):
            break

        imgOut, roi, face_detected = GetFaceFromVideo(vidFrame)
        #cv.imshow("Roi", roi)
        if(face_detected):
            #Dimensions of Region of Interest
            roi_width = roi.shape[1]
            roi_half_width = math.ceil(roi_width/2)
            roi_height = roi.shape[0]
            roi_half_height = math.ceil(roi_height/2)
            
            #Begin Sobel in openCl
            roiBGRA = cv.cvtColor(roi, cv.COLOR_BGR2BGRA)
            roiBGRACopy = roiBGRA.copy()
            imgFormat = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)

            bufferFilterIn = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format=imgFormat, shape=(roi_width, roi_height), pitches=(roiBGRA.strides[0], roiBGRA.strides[1]), hostbuf=roiBGRA.data)
            bufferFilterOut = cl.Image(ctx, flags=cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, format = imgFormat, shape=(roi_width, roi_height), pitches=(roiBGRACopy.strides[0], roiBGRACopy.strides[1]), hostbuf=roiBGRACopy.data)

            kernel = prg.sobel_and_filtering
            kernel.set_arg(0, bufferFilterIn)
            kernel.set_arg(1, np.int32(roi_width))
            kernel.set_arg(2, np.int32(roi_height))
            kernel.set_arg(3, np.int32(diff))
            kernel.set_arg(4, np.int32(average))
            kernel.set_arg(5, bufferFilterOut)

            workGroupSize = (math.ceil(roi_width/32)*32, math.ceil(roi_height/32)*32)
            workItemSize = (32,32)

            kernelEvent = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size = workGroupSize, local_work_size = workItemSize)
            kernelEvent.wait()

            cl.enqueue_copy(queue, roiBGRACopy.data, bufferFilterOut, origin = (0,0), region = (roi_width, roi_height), is_blocking=True)

            bufferFilterIn.release()
            bufferFilterOut.release() 
            #cv.imshow("Sobel", roiBGRACopy)
            roiLeft = roiBGRACopy[0:roi_half_height, 0:roi_half_width].copy()
            roiRight = roiBGRACopy[0:roi_half_height, roi_half_width:roi_width].copy()
            #cv.imshow("Sobel Right", roiRight)
            #cv.imshow("Sobel Left", roiLeft)
            #End Sobel
            
            #Eye Detection in openCL
            accumulator = np.zeros((roi_half_width + roi_half_height * roi_half_width + (raio_max - raio_min) * roi_half_width * roi_half_height), dtype=np.int32)
            circle = np.zeros(3, dtype=np.int32)
            maxval = np.int32(0)

            #Left Eye
            roiLeftCopy = roiLeft.copy()
            bufferFilterIn = cl.Image(ctx, flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, format=imgFormat, shape=(roi_half_width, roi_half_height), pitches=(roiLeft.strides[0], roiLeft.strides[1]), hostbuf=roiLeft.data)
            bufferMaxVal = cl.Buffer(ctx, flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = maxval)
            bufferAccumulator = cl.Buffer(ctx, flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = accumulator)
            bufferCircle = cl.Buffer(ctx, flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = circle)

            kernel = prg.circleHough
            kernel.set_arg(0, bufferFilterIn)
            kernel.set_arg(1, np.int32(roi_half_width))
            kernel.set_arg(2, np.int32(roi_half_height))
            kernel.set_arg(3, np.int32(raio_min))
            kernel.set_arg(4, np.int32(raio_max))
            kernel.set_arg(5, bufferMaxVal)
            kernel.set_arg(6, bufferAccumulator)
            kernel.set_arg(7, bufferCircle)

            kernelEvent = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size = workGroupSize, local_work_size = workItemSize)
            kernelEvent.wait()
            
            cl.enqueue_copy(queue, circle.data, bufferCircle)
            bufferFilterIn.release()
            bufferMaxVal.release()
            bufferAccumulator.release()
            bufferCircle.release()

            """
            bufferFilterOut = cl.Image(ctx, flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, format = imgFormat, shape=(roi_half_width, roi_height), pitches=(roiLeftCopy.strides[0], roiLeftCopy.strides[1]), hostbuf=roiLeftCopy.data)
            bufferCircle = cl.Buffer(ctx, flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = circle)
            kernel = prg.desingCircle
            kernel.set_arg(0, np.int32(roi_half_width))
            kernel.set_arg(1, np.int32(roi_height))
            kernel.set_arg(2, bufferCircle)
            kernel.set_arg(3, bufferFilterOut)

            kernelEvent = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size = workGroupSize, local_work_size = workItemSize)
            kernelEvent.wait()

            cl.enqueue_copy(queue, roiLeftCopy.data, bufferFilterOut, origin = (0,0), region = (roi_half_width, roi_height), is_blocking=True)
            print(circle)
            print(roi_half_width)
            print(roi_height)
            bufferFilterOut.release() 
            bufferCircle.release() 
            """

            #cv.imshow("Left", roiLeft)
            #cv.imshow("Left Eye", roiLeftCopy)
            pt1 = (circle[0] - circle[2] - 3, circle[1] - circle[2] - 3) 
            pt2 = (circle[0] + circle[2] + 3, circle[1] + circle[2] + 3)
            roi = cv.rectangle(roi, pt1, pt2, (0, 255, 0, 255), 2)

            accumulator = np.zeros((roi_half_width + roi_half_height * roi_half_width + (raio_max - raio_min) * roi_half_width * roi_half_height), dtype=np.int32)
            circle = np.zeros(3, dtype=np.int32)
            maxval = np.int32(0)

            #Right Eye
            roiRightCopy = roiRight.copy()
            bufferFilterIn = cl.Image(ctx, flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, format=imgFormat, shape=(roi_width - roi_half_width, roi_half_height), pitches=(roiRight.strides[0], roiRight.strides[1]), hostbuf=roiRight.data)
            bufferMaxVal = cl.Buffer(ctx, flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = maxval)
            bufferAccumulator = cl.Buffer(ctx, flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = accumulator)
            bufferCircle = cl.Buffer(ctx, flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = circle)

            kernel = prg.circleHough
            kernel.set_arg(0, bufferFilterIn)
            kernel.set_arg(1, np.int32(roi_width - roi_half_width))
            kernel.set_arg(2, np.int32(roi_half_height))
            kernel.set_arg(3, np.int32(raio_min))
            kernel.set_arg(4, np.int32(raio_max))
            kernel.set_arg(5, bufferMaxVal)
            kernel.set_arg(6, bufferAccumulator)
            kernel.set_arg(7, bufferCircle)

            kernelEvent = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size = workGroupSize, local_work_size = workItemSize)
            kernelEvent.wait()
            
            cl.enqueue_copy(queue, circle.data, bufferCircle)
            bufferFilterIn.release()
            bufferMaxVal.release()
            bufferAccumulator.release()
            bufferCircle.release()

            """
            bufferFilterOut = cl.Image(ctx, flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, format = imgFormat, shape=(roi_width - roi_half_width, roi_height), pitches=(roiRightCopy.strides[0], roiRightCopy.strides[1]), hostbuf=roiRightCopy.data)
            bufferCircle = cl.Buffer(ctx, flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = circle)
            kernel = prg.desingCircle
            kernel.set_arg(0, np.int32(roi_width - roi_half_width))
            kernel.set_arg(1, np.int32(roi_height))
            kernel.set_arg(2, bufferCircle)
            kernel.set_arg(3, bufferFilterOut)

            kernelEvent = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size = workGroupSize, local_work_size = workItemSize)
            kernelEvent.wait()

            cl.enqueue_copy(queue, roiRightCopy.data, bufferFilterOut, origin = (0,0), region = (roi_width - roi_half_width, roi_height), is_blocking=True)
            print(circle)
            print(roi_half_width)
            print(roi_height)
            bufferFilterOut.release() 
            bufferCircle.release() 
            """

            #cv.imshow("Right", roiRight)
            #cv.imshow("Right Eye", roiRightCopy)
            pt1 = (roi_half_width + circle[0] - circle[2] - 3, circle[1] - circle[2] - 3) 
            pt2 = (roi_half_width + circle[0] + circle[2] + 3, circle[1] + circle[2] + 3)
            roi = cv.rectangle(roi, pt1, pt2, (0, 255, 0, 255), 2)
            #Fim

        print("- execute --- %s seconds ---" % (1/(time.time() - start_time)))
        
        cv.imshow("Video", imgOut)

        if (cv.waitKey(20) >= 0):
            break
