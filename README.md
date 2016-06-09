# Label ignore/filter for torch/nn

Ignore/bypass some inputs according to a specific set of ignore labels. 
This package allows for generally available, out-of-the-box criterions in [torch/nn](https://github.com/torch/nn)
to be used in situations where a specific label of a certain class/output 
is needed to be overlooked/ignored when backproping through a network model.


## Usage

This package consists of two submodules: 

- `criterion_filter.Single(Criterion, Ignore_Label)` wraps an input criterion and filters all indexes which have the same label as in `Ignore_Label`. 
- `criterion_filter.Parallel()` is a direct refactoring of the `nn.ParallelCriterion()` container, where `:add(Criterion, Weight, Ignore_label)` allows you to specify a set of ignore labels for each criterion that you add.

The following code snippets shows how to use the package for cases using single and multiple criterias.

### Single criterion

Example 1
```lua
--[[ Example using `ClassNLLCriterion`. 
Computes the loss for an input tensor with all allowed targets,
and proceed to compute the loss with a blacklisted label. 
]]
require 'criterion_filter'

-- 1. define model
model = nn.Sequential()
model:add(nn.Linear(10,4))

-- 2. define criterion
nll = nn.ClassNLLCriterion()
criterion = criterion_filter.Single(nll, 0) -- set to ignore class 0

-- 3. define input data 
input = torch.Tensor(5,10):uniform()
target = torch.Tensor(5):random(1,4)

-- 4. compute loss (forward pass only)
-- 4.1. (no labels to be ignored at this point)
output = model:forward(input)
print(output)
err1 = criterion:forward(output,target)
print(err1)

-- 4.2. set one target label to 0 (to be ignored)
target[5] = 0
err2 = criterion:forward(output,target)
print(err2)
print(err1 == err2)
```
Example 2
```lua
--[[ Example using `MSECriterion`.
Computes the loss and gradient for an input tensor + targets,
and proceeds to compute the loss and gradient with blacklisted labels.
]]
require 'criterion_filter'

-- 1. define model
model = nn.Sequential()
model:add(nn.Linear(10,4))

-- 2. define criterion
mse = nn.MSECriterion()
criterion = criterion_filter.Single(mse, {torch.Tensor(4):fill(0), {{1,2,3,4}}}) -- set to ignore class two labels

-- 3. define input data 
input = torch.Tensor(5,10):uniform()
target = torch.Tensor(5,4):uniform()

-- 4. compute loss+gradient (forward+backward pass)
-- 4.1. (no labels to be ignored at this point)
output = model:forward(input)
err1 = criterion:forward(output,target)
gradOutput1 = criterion:backward(output,target)
print(err1)
print(gradOutput1)

-- 4.2. set one target to 0's (to be ignored)
target[1]:fill(0)
err2 = criterion:forward(output,target)
gradOutput2 = criterion:backward(output,target)
print(err2)
print(gradOutput2)

--4.3. set another target to 0's and one to {1,2,3,4} 
target[3]:fill(0)
target[4]:copy(torch.Tensor{1,2,3,4})
err3 = criterion:forward(output,target)
gradOutput3 = criterion:backward(output,target)
print(err3)
print(gradOutput3)
```

## Multiple criterions

Example 1
```lua
--[[ Example using multiple criterions (`ClassNLLCriterion` + `ClassNLLCriterion`).
Here, the loss is computed using two equal criterias with equal weigths. 
Check how the loss value differs when it is computed with ignored/filtered/blacklisted labels.
]]
require 'criterion_filter'

-- 1. define model
model = nn.Sequential()
prl = nn.ConcatTable()
prl:add(nn.Linear(10,10))
prl:add(nn.Linear(10,10))
model:add(prl)

-- 2. define criterions
criterion = criterion_ignore.Parallel()
criterion:add(nn.ClassNLLCriterion(), 1, 6) -- set different ignore labels
criterion:add(nn.ClassNLLCriterion(), 1, 7) -- set different ignore labels

-- 3. define input data
input = torch.Tensor(5,10):uniform()
target1 = torch.Tensor(5):random(1,10)
target2 = torch.Tensor(5):random(1,10)

-- 4. compute loss
-- 4.1. without ignore labels
output = model:forward(input)
print(output[1])
print(output[2])
print(target1)
print(target2)
err1 = criterion:forward(output,{target1, target2})
print(err1)

-- 4.2. now with ignored/blacklisted labels
target1[2] = 6
target2[2] = 7 
err2 = criterion:forward(output,{target1, target2})
print(err2)

-- 4.3. flip labels 
target1[2] = 7
target2[2] = 6 
err3 = criterion:forward(output,{target1, target2})
print(err3)
```

Example 2

```lua
--[[ Example using multiple criterions (`ClassNLLCriterion` + `MSECriterion`).
Here, the loss is computed using two different criterias and weigths. 
Check how the loss value differs when it is computed with ignored/filtered/blacklisted labels.
]]
require 'criterion_filter'

-- 1. define model
model = nn.Sequential()
prl = nn.ConcatTable()
prl:add(nn.Linear(10,5))
prl:add(nn.Linear(10,4))
model:add(prl)

-- 2. define criterions
criterion = criterion_ignore.Parallel()
criterion:add(nn.ClassNLLCriterion(), 1)  -- no ignore label defined
criterion:add(nn.MSECriterion(), 0.5, torch.Tensor({1,1,1,1})) -- set an ignore label

-- 3. define input data
input = torch.Tensor(5,10)
target1 = torch.Tensor(5):random(1,5)
target2 = torch.Tensor(5,4):uniform()

-- 4. compute loss
-- 4.1. without ignore labels
output = model:forward(input)
err1 = criterion:forward(output, {target1, target2})
print(err1)

-- 4.2. now with ignored/blacklisted labels
target2[2]:fill(1)
target2[5]:fill(1)
err2 = criterion:forward(output, {target1, target2})
print(err2)
```
