from osgeo import gdal, ogr, gdalnumeric
import tifffile as tiff
import numpy as np
import os

###############################################
###### 2 functions from Emanuele Dalasso ######
###############################################

def DistanceTransform(rasterSrc, vectorSrc, npDistFileName='', units='pixels'):
    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize
    noDataValue = 0

    if units == 'meters':
        geoTrans, poly, ulX, ulY, lrX, lrY = gT.getRasterExtent(srcRas_ds)
        transform_WGS84_To_UTM, transform_UTM_To_WGS84, utm_cs = gT.createUTMTransform(poly)
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(geoTrans[0], geoTrans[3])
        line.AddPoint(geoTrans[0] + geoTrans[1], geoTrans[3])

        line.Transform(transform_WGS84_To_UTM)
        metersIndex = line.Length()
    else:
        metersIndex = 1

    ## create First raster memory layer
    memdrv = gdal.GetDriverByName('MEM')
    dst_ds = memdrv.Create('', cols, rows, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)

    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[255])
    srcBand = dst_ds.GetRasterBand(1)

    memdrv2 = gdal.GetDriverByName('MEM')
    prox_ds = memdrv2.Create('', cols, rows, 1, gdal.GDT_Int16)
    prox_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    prox_ds.SetProjection(srcRas_ds.GetProjection())
    proxBand = prox_ds.GetRasterBand(1)
    proxBand.SetNoDataValue(noDataValue)
    options = ['NODATA=0']

    ##compute distance to non-zero pixel values and scrBand and store in proxBand
    gdal.ComputeProximity(srcBand, proxBand, options)

    memdrv3 = gdal.GetDriverByName('MEM')
    proxIn_ds = memdrv3.Create('', cols, rows, 1, gdal.GDT_Int16)
    proxIn_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    proxIn_ds.SetProjection(srcRas_ds.GetProjection())
    proxInBand = proxIn_ds.GetRasterBand(1)
    proxInBand.SetNoDataValue(noDataValue)
    options = ['NODATA=0', 'VALUES=0']

    ##compute distance to zero pixel values and scrBand and store in proxInBand
    gdal.ComputeProximity(srcBand, proxInBand, options)

    proxIn = gdalnumeric.BandReadAsArray(proxInBand)
    proxOut = gdalnumeric.BandReadAsArray(proxBand)

    ##distance tranform is the distance to zero pixel values minus distance to non-zero pixel values
    proxTotal = proxIn.astype(float) - proxOut.astype(float)
    proxTotal = proxTotal * metersIndex

    if npDistFileName != '':
        np.save(npDistFileName, proxTotal)

    return proxTotal

def CreateClassSegmentation(rasterSrc, vectorSrc, npDistFileName='', units='pixels'):
    dist_trans = DistanceTransform(rasterSrc, vectorSrc, npDistFileName='', units='pixels')
    dist_trans[dist_trans > 0] = 1
    dist_trans[dist_trans < 0] = 0
    return dist_trans

def norma_sep(im):
    # Ignores zeros
    ima = im.copy()
    for i in range(im.shape[2]):
      im_no_zero = im[:,:,i].ravel()[im[:,:,i].ravel() != 0] 
      im_norm = ((im[:,:,i] - im_no_zero.min())*255. / (im[:,:,i].max() - im_no_zero.min())).astype(np.uint8)
      im_norm[im[:,:,i] == 0] = 0
      ima[:,:,i] = im_norm
    return ima.astype(np.uint8)
