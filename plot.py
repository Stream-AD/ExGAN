from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt

latcorners = np.array([23.476929, 20.741224, 45.43908 , 51.61555 ])
loncorners = np.array([-118.67131042480469, -82.3469009399414,
                   -64.52022552490234, -131.4470977783203])
lon_0 = -105
lat_0 = 60

def plot_precip(data):
	'''
	data is a 813*1051 matrix containing unnormalized precipitation values
	'''
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
	            llcrnrlat=latcorners[0],urcrnrlat=latcorners[2],\
	            llcrnrlon=loncorners[0],urcrnrlon=loncorners[2],\
	            rsphere=6371200.,resolution='i', area_thresh=10000)
	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()
	m.drawlsmask(land_color="#FCF8F3", ocean_color='#E6FFFF')
	parallels = np.arange(0.,90,10.)
	m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
	meridians = np.arange(180.,360.,10.)
	m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
	ny = data.shape[0]; nx = data.shape[1]
	lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
	x, y = m(lons, lats) # compute map proj coordinates.
	clevs = np.array([0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750])
	cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
	cbar = m.colorbar(cs,location='bottom',pad="5%")
	cbar.set_label('mm')
	plt.show()