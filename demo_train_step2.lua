require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'image'
local optim = require 'optim'
local mat = require('fb.mattorch')

local function cudnnize(model)
    model:cuda()
    cudnn.convert(model, cudnn)
end

optimState = {
    learningRate = 0.1,
    learningRateDecay = 0.0,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0,
    weightDecay = 1e-4,
}
local batch_size = 1

local depth = image.load('input/depth.png', 3)[{{1}, {}, {}}]:view(batch_size, 1, 256, 256):mul(100)
local normal = image.load('input/normal.png'):view(batch_size, 3, 256, 256):mul(100)
local input = torch.cat(normal, depth, 2):cuda()

local target = mat.load('input/voxels.mat').voxels
target[torch.le(target, 0.5)] = 0.0
target[torch.gt(target, 0.5)] = 1.0
target = target:cuda()

local step2 = torch.load('models/step2.t7'):add(nn.Reshape(128, 128, 128, true))
local criterion = nn.BCECriterion()
criterion:cuda()

cudnnize(step2)
params, gradParams = step2:getParameters()

local function feval()
    return criterion.output, gradParams
end

local output = step2:forward(input)
local loss = criterion:forward(output, target)
print('loss = ' .. loss)

step2:zeroGradParameters()
criterion:backward(output, target)
step2:backward(input, criterion.gradInput)

optim.sgd(feval, params, optimState)
