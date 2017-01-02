local ImgJitter = {}

ImgJitter.getTransform = function (config)
  local c00=1
  local c01=0
  local c10=0
  local c11=1

  if config.xStretch then
	  c00 = c00*config.xStretch
  end

  if config.yStretch then
	  c11 = c11*config.yStretch
  end

  if config.hFlip then
	c00 = -c00 --Horizontal flip
  end

  if config.vFlip then
	c11 = -c11 --Horizontal flip
  end

  if config.xShear then
	  c01=config.xShear*c00 + c01
	  c11=config.xShear*c10  + c11
  end

  if config.yShear then
	  c10=config.yShear*c00 + c10
	  c11=config.yShear*c01 + c11
  end

  if config.rotate then
	local theta = config.rotate
    local c=math.cos(theta) 
    local s=math.sin(theta)
    local t00=c00*c-c01*s 
	local t01=c00*s+c01*c 
	c00=t00
	c01=t01
    local t10=c10*c-c11*s 
	local t11=c10*s+c11*c 
	c10=t10
	c11=t11
  end

  --compute the coordinates of frames after transformation
  local trMat = torch.Tensor({ {c00, c01}, {c10, c11}})
  local srcX = config.srcWidth-1
  local srcY = config.srcHeight-1
  local src = torch.Tensor({{0,0}, {srcX, 0}, {0, srcY}, {srcX, srcY}})
  local dst = torch.mm(src, trMat)
  --minimum coordinates, maximum coordinates
  local minDst = dst:min(1):squeeze()
  local maxDst = dst:max(1):squeeze()

  local dstSz = maxDst - minDst
  dstSz:add(1)
  local dstWidth = dstSz[1]
  local dstHeight = dstSz[2]
  --print ("dst size")
  --print (dstSz)

  local outWidth = config.outWidth or config.srcWidth
  local outHeight = config.outHeight or config.srcHeight

  local xCrop = config.xCrop or outWidth/2
  local yCrop = config.yCrop or outHeight/2

  local dstRoiX = dstWidth/2 -  xCrop
  local dstRoiY = dstHeight/2 -  yCrop

  dstRoiX = math.max(dstRoiX, 0)
  dstRoiY = math.max(dstRoiY, 0)

  --assert(dstRoiX + outWidth <= dstWidth, "ROI out of range")
  --assert(dstRoiY + outHeight <= dstHeight, "ROI out of range")
  local coeff = torch.Tensor({ {c00, c10, -minDst[1]-dstRoiX }, {c01, c11, -minDst[2]-dstRoiY}})
	--print (coeff)
  return coeff
end

ImgJitter.getJDTransform = function (srcWidth, srcHeight, outWidth, outHeight, crpStart, crpEnd, centerOnly)
	local szRatioX = outWidth/srcWidth
	local szRatioY = outHeight/srcHeight
	local crpStart = crpStart or 0.7
	local crpEnd = crpEnd or 0.9
	local centerOly = centerOly or false

	local hFlip = math.random() < 0.5
	local crpSz = torch.uniform(crpStart, crpEnd)
	local xOffset = (1.0 - crpSz)/2
	local yOffset = (1.0 - crpSz)/2

	if not centerOnly then
		xOffset = torch.uniform(0, 1.0 - crpSz)
		yOffset = torch.uniform(0, 1.0 - crpSz)
	end

	local xStretch = szRatioX/crpSz
    local yStretch = szRatioY/crpSz
	local xCrop = (0.5 - xOffset)*srcWidth*xStretch --distance to the center
	local yCrop = (0.5 - yOffset)*srcHeight*yStretch
	--print ("x " ..xCrop)
	--print ("y ".. yCrop)
	local rotation = 4 * math.random()
	local config = {
						srcWidth = srcWidth,
						srcHeight = srcHeight,
                        rotate = rotation * 2 * math.pi / 4,
                        hFlip = hFlip,
						xStretch = xStretch,
						yStretch = yStretch,
                        outWidth = outWidth ,
                        outHeight = outHeight,
						xCrop = xCrop,
						yCrop = yCrop
					}
    return ImgJitter.getTransform(config)	
end

ImgJitter.getJDTransform2 = function (srcWidth, srcHeight, outWidth, outHeight, crpStart, crpEnd, centerOnly)
	local szRatioX = outWidth/srcWidth
	local szRatioY = outHeight/srcHeight
	local crpStart = crpStart or 0.7
	local crpEnd = crpEnd or 0.9
	local centerOly = centerOly or false

	local hFlip = math.random() < 0.5
	local crpSzX = torch.uniform(crpStart, crpEnd)
	local crpSzY = torch.uniform(crpStart, crpEnd)
	local xOffset = (1.0 - crpSzX)/2
	local yOffset = (1.0 - crpSzY)/2

	if not centerOnly then
		xOffset = torch.uniform(0, 1.0 - crpSzX)
		yOffset = torch.uniform(0, 1.0 - crpSzY)
	end

	local xStretch = szRatioX/crpSzX
    local yStretch = szRatioY/crpSzY
	local xCrop = (0.5 - xOffset)*srcWidth*xStretch --distance to the center
	local yCrop = (0.5 - yOffset)*srcHeight*yStretch
	--print ("x " ..xCrop)
	--print ("y ".. yCrop)
	local rotation = 2 * math.random()*math.pi
	local config = {
						srcWidth = srcWidth,
						srcHeight = srcHeight,
                        rotate = rotation,
                        hFlip = hFlip,
						xStretch = xStretch,
						yStretch = yStretch,
                        outWidth = outWidth ,
                        outHeight = outHeight,
						xCrop = xCrop,
						yCrop = yCrop
					}
    return ImgJitter.getTransform(config)	
end

ImgJitter.getEmbedTransform = function (srcWidth, srcHeight, outWidth, outHeight, embedStart, embedEnd, centerOnly)
	local szRatioX = outWidth/srcWidth
	local szRatioY = outHeight/srcHeight
	local embedStart = embedStart or 0.7
	local embedEnd = crpEnd or 0.9
	local centerOly = centerOly or false

	local hFlip = math.random() < 0.5
	local embedSz = torch.uniform(embedStart, embedEnd)
	local xOffset = (1.0 - embedSz)/2
	local yOffset = (1.0 - embedSz)/2

	if not centerOnly then
		xOffset = torch.uniform(0, 1.0 - embedSz)
		yOffset = torch.uniform(0, 1.0 - embedSz)
	end

	local xStretch = szRatioX*embedSz
    local yStretch = szRatioY*embedSz
	local xCrop = (0.5 - xOffset)*outWidth --distance to the center
	local yCrop = (0.5 - yOffset)*outHeight
	--print ("x " ..xCrop)
	--print ("y ".. yCrop)
	local rotation = math.random()*2*math.pi
	local config = {
						srcWidth = srcWidth,
						srcHeight = srcHeight,
                        rotate = rotation,
                        hFlip = hFlip,
						xStretch = xStretch,
						yStretch = yStretch,
                        outWidth = outWidth ,
                        outHeight = outHeight,
						xCrop = xCrop,
						yCrop = yCrop
					}
    return ImgJitter.getTransform(config)	
end

function ImgJitter.test()
	for i = 1, 10 do
		--local coeff,roi = imgJitter.getJDTransform(256, 256, 128, 128)
		local coeff = imgJitter.getJDTransform(512, 512, 256, 256)
		print (coeff)
	end
end

--imgJitter.test()
return ImgJitter
