require 'image'

local t = require 'transforms'
local ffi = require 'ffi'
local Model = require 'model'
local mat = require('fb.mattorch')

function file_exists(name)
	local f = io.open(name,"r")
	if f ~= nil then
		io.close(f)
		return true
	else
		return false
	end
end

function loadImage(path)
	local ok, img = pcall(function()
		return image.load(path, 3)
	end)
	-- Sometimes image.load fails because the file extension does not match the
	-- image format. In that case, use image.decompress on a ByteTensor.
	if not ok then
		local f = io.open(path, 'r')
		assert(f, 'Error reading: ' .. tostring(path))
		local data = f:read('*a')
		f:close()
		local b = torch.ByteTensor(string.len(data))
		ffi.copy(b:data(), data, b:size(1))
		img = image.decompress(b, 3)
	end
	return img
end

function preprocess()
	return t.Compose{
		t.Scale(opt.imgDim),
		t.ColorNormalize(opt.meanstd),
	}
end

function findLast(haystack, needle)
	local i = haystack:match(".*"..needle.."()")
	if i == nil then
		return nil
	else
		return i - 1
	end
end

cmd = torch.CmdLine()
cmd:option('-imgname', '', 'The name of test image, which should be stored in "image" folder.')
opt = cmd:parse(arg or {})
opt.imgpath = paths.concat('image', opt.imgname)
opt.imgDim = 256
opt.www = 'output/'
opt.meanstd = {
	mean = { 0.485, 0.456, 0.406 },
	std = { 0.229, 0.224, 0.225 },
}

assert(file_exists(opt.imgpath), "Test image: '" .. opt.imgpath .. "' does not exist.")

if not file_exists(opt.www) then
	os.execute("mkdir " .. opt.www)
end

local img = torch.Tensor(1, 3, 256, 256)
img[1] = image.scale(preprocess()(loadImage(opt.imgpath)), 256, 256)

local model = Model()
output = model:test(img)
local savename = string.sub(opt.imgname, 1, findLast(opt.imgname, '%.')) .. 'mat'
savepath = sys.concat(opt.www, savename)
mat.save(savepath, {['voxels'] = output})

