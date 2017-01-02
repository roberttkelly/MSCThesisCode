require 'paths'
require 'image'
--simple script to crop the images

torch.setdefaulttensortype('torch.FloatTensor')

function embed(im, width, height, defaultValue)
    defaultValue = defaultValue or im:mean()
    local tmpImg = torch.Tensor(im:size(1), height, width):fill(defaultValue)
    local offset1 = math.floor((tmpImg:size(2) - im:size(2))/2)+1
    local offset2 = math.floor((tmpImg:size(3) - im:size(3))/2)+1
    tmpImg:narrow(2, offset1, im:size(2)):narrow(3, offset2, im:size(3)):copy(im)
    return tmpImg
end

function scaleImg(im, sz, strech, defaultValue)
   stretch = stretch or true --whether to stretch smaller images
   local width = im:size(3)
   local height = im:size(2)

   if ((not stretch) and width < sz[3] and height < sz[2]) then
       return embed(im, sz[3], sz[2], defaultValue)
   end

   defaultValue = defaultValue or im:mean()
   local im1 = torch.Tensor(sz[1], sz[3], sz[2]):fill(defaultValue)
   if (width > height) then
       local newH = math.floor(sz[3]*height/width)
       im = image.scale(im, sz[2], newH)
       local offset = math.floor((sz[3] - newH)/2) +1
       im1:narrow(2,offset,newH):copy(im)
   else
       local newW = math.floor(sz[2]*width/height)
       im = image.scale(im, newW, sz[3])
       local offset = math.floor((sz[2] - newW)/2) + 1
       im1:narrow(3,offset,newW):copy(im)
   end
   return im1
end


function string:split( inSplitPattern, outResults ) 
  outResults = outResults or {} 
  local ind =1 
  local theStart = 1 
  local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart ) 
  while theSplitStart do 
    outResults[ind] =  string.sub( self, theStart, theSplitStart-1 )  
    ind = ind + 1 
    theStart = theSplitEnd + 1 
    theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart ) 
  end 
  outResults[ind] = string.sub( self, theStart )  
  return outResults 
end 

function getVectorBound(x, thr, margin)
	--thr = thr or 0.08
	--thr = x:max()/10
	local l = x:size(1)
	thr = x[{{l/3, 2*l/3}}]:mean()/15
	--thr = math.min(thr, thr1)
	margin = margin or 10 
	local min = -1
	local len = x:size(1)
	local max = -1 
	for i=1,len do
	   if (x[i] > thr) then
	       local ex = i+ margin
		   ex = math.min(ex, len)
	       local found = true
	       for j=i+1,ex do 
			   --print (x[j])
		      if x[j] < thr then
			     found = false
				 break
			  end
		   end
		   if found then 
		      min = i
			  break
		   end
	   end
    end

	for i=len,1,-1 do
	   if (x[i] > thr) then
	       local ex = i - margin
		   ex = math.max(ex, 1)
	       local found = true
	       for j=i-1,ex,-1 do 
			   --print (x[j])
		      if x[j] < thr then
			     found = false
				 break
			  end
		   end
		   if found then 
		      max = i
			  break
		   end
	   end
   end
   return min,max
end


function getMaskRegion(im)
    im = im[1]
    --print (im:size())
    local max_x = im:max(1)
    local max_y = im:max(2)
    local x1,x2 = getVectorBound(max_x:resize(max_x:nElement()))
    local y1,y2 = getVectorBound(max_y:resize(max_y:nElement()))
    --print (x1, x2)
    --print (y1, y2)
    assert( x1>0 and x2>0, "cannot find mask region for x bounds\n")
    assert( y1>0 and y2>0, "cannot find mask region for y bounds\n")
    return x1,y1, x2, y2
end


local inFileList = arg[1]
local outDir = arg[2]
local sampleSize = {3, tonumber(arg[3]),tonumber(arg[3])}
local labels = {}
local fList = {}
for fname in io.lines(inFileList) do
	table.insert(fList, fname)
end

for i = 1,#fList do
   local imgName = fList[i]

   local tmp = imgName:split("/")
   local im = image.load(imgName)
   local sz = im:size()
   local mask = image.rgb2yuv(im)
   x1,y1, x2, y2 = getMaskRegion(mask)
   im1 = image.crop(im, x1, y1, x2, y2)
   im1 = scaleImg(im1, sampleSize, true, 0)
   image.saveJPG(arg[2].."/"..tmp[#tmp],  im1)
   print (string.format("%s %d %d %d %d", tmp[#tmp], x1, y1, x2, y2))
   collectgarbage()
end
