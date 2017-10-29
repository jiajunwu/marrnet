--  Modified from https://github.com/facebook/fb.resnet.torch
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.Compose(transforms)
  return function(input)
    for _, transform in ipairs(transforms) do
      input = transform(input)
    end
    return input
  end
end

function M.ColorNormalize(meanstd)
  return function(img)
    img = img:clone()
    for i = 1, 3 do
      img[i]:add(-meanstd.mean[i])
      img[i]:div(meanstd.std[i])
    end
    return img
  end
end


-- Scales the shorter/longer edge to size
function M.Scale(size, minmax, interpolation)
  interpolation = interpolation or 'bicubic'
  return function(input)
    local isMax = minmax == 'max'
    local w, h = input:size(input:dim()), input:size(input:dim() - 1)
    if (not isMax and (w <= h and w == size) or (h <= w and h == size)) or
      (isMax and (w >= h and w == size) or (h >= w and h == size)) then
      return input
    end
    if (w < h and not isMax) or (w > h and isMax)  then
      return image.scale(input, size, h/w * size, interpolation)
    else
      return image.scale(input, w/h * size, size, interpolation)
    end
  end
end

return M
