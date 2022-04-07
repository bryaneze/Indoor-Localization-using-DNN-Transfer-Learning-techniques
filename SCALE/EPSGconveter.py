from pyproj import Proj, transform #converting from EPSG4326 to EPSG3414 
import pandas as pd 
df = pd.read_csv('./17-Aug-data-collection.csv')
inProj = Proj(init='epsg:4326') #initial format 
outProj = Proj(init='epsg:3414') #output format 
temp_lat = [] 
temp_long = [] 
for i in range(len(df)): 
    x,y = transform(inProj,outProj,df['longitude'].iloc[i],df['latitude'].iloc[i]) 
    temp_lat.append(y) 
    temp_long.append(x)


df['latitude'] = temp_lat 
df['longitude'] = temp_long

df.to_csv('new.csv')

