require 'nn'
require 'cudnn'
require 'cutorch'
local M = {}
local Model = torch.class('Model', M)

local function cudnnize(model)
	model:cuda()
	cudnn.convert(model, cudnn)
end

function Model:__init()
	self.step1 = torch.load('models/step1.t7')
	self.step2 = torch.load('models/step2.t7'):add(nn.Reshape(128, 128, 128, true))
	cudnnize(self.step1)
	cudnnize(self.step2)
end

local function mask(input)
	local normal = input[1]
	local depth = input[2]
	local sil = input[3][{{}, 1, {}, {}}]
	for	i = 1, 3 do
		normal[{{}, i, {}, {}}][torch.le(sil, 30)] = 100
	end
	depth[{{}, 1, {}, {}}][torch.le(sil, 30)] = 0
	return torch.cat(normal, depth, 2)
end

function Model:test(img)
	local input = torch.CudaTensor()
	input:resize(img:size()):copy(img)
	local step1_out = self.step1:forward(input)
	input = mask(step1_out)
	local output = self.step2:forward(input)
	return output:double()
end

return M.Model

