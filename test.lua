--[[
    Test scripts.
]]

local mytest = torch.TestSuite()

local tester = torch.Tester()

local precision = 1e-5

torch.manualSeed(4)

require 'criterion_filter'
--[[
-- for debug purposes only
require 'nn'
criterion_filter = {}
paths.dofile('src/SingleCriterionFilter.lua')
paths.dofile('src/ParallelCriterionFilter.lua')
-]]

--===================================================
-- Single Criterion
--===================================================

function mytest.SingleFwdBwdClassNLLCriterionNoIgnoreLabel()
-- Do a forward pass through ClassNLLCriterion and it should give the same error even if the tensor is shuffled.

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local criterion = criterion_filter.Single(nll, 1) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(1,5)
    local err1 = criterion:forward(input,target)
    
    -- shuffle indexes
    local shuffledIdx = torch.randperm(10):long()
    local err2 = criterion:forward(input:index(1,shuffledIdx),target:index(1,shuffledIdx))

    tester:eq(err1, err2, "Forward: err1 and err2 should be equal")
    
    local grad = criterion:backward(input, target)
    
    tester:assertTensorNe(grad, grad:clone():fill(0), "Backward: gradient should be different than 0")
end

function mytest.SingleFwdClassNLLCriterionIgnoreLabel()
-- Do a forward pass through ClassNLLCriterion and it should give different results

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local criterion = criterion_filter.Single(nll, 5) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(1,4)
    local err1 = criterion:forward(input,target)
    
    -- shuffle indexes
    target[1]=5 -- force this value to at least be 5
    local err2 = criterion:forward(input,target)

    tester:ne(err1, err2, "err1 and err2 should be different")
end

function mytest.SingleFwdBwdClassNLLCriterionIgnoreLabelALL()
-- Do a forward pass through ClassNLLCriterion and it should give 0.

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local criterion = criterion_filter.Single(nll, 5) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):fill(5)
    local err1 = criterion:forward(input,target)
    
    tester:eq(err1, 0, "err1 should be equal to 0")
    
    local grad = criterion:backward(input, target)
    
    tester:assertTensorEq(grad, grad:clone():fill(0), precision, "Backward: gradient should be equal to 0")
    tester:assertTensorEq(torch.LongTensor(grad:size()), torch.LongTensor(input:size()), "Backward: gradient and input size should be the same.")
end

function mytest.SingleFwdMSECriterionNoIgnoreLabel()
-- Do a forward pass through MSECriterion and it should give the same error even if the tensor is shuffled.

    -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, -1) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10,5):uniform()
    local err1 = criterion:forward(input,target)
    
    -- shuffle indexes
    local shuffledIdx = torch.randperm(10):long()
    local err2 = criterion:forward(input:index(1,shuffledIdx),target:index(1,shuffledIdx))

    tester:eq(err1,err1, precision, "err1 should be equal to err2")
end

function mytest.SingleFwdMSECriterionIgnoreLabel()
-- Do a forward pass through MSECriterion and it should give different results 

    -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, 0) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10,5):uniform()
    local err1 = criterion:forward(input,target)
    
    -- shuffle indexes
    input[1]:fill(0)-- force this value to at least be 5
    local err2 = criterion:forward(input,target)
    
    tester:assertgt(math.abs(err1-err2), precision, "err1-err2 should be greater than " .. precision)
end

function mytest.SingleFwdMSECriterionIgnoreSameError()
-- Do a forward pass through MSECriterion and it should give different results 

    -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, 0) -- set to ignore class 0

    local input = torch.Tensor(10,5):fill(1)
    input[1]:fill(1)
    local target = torch.Tensor(10,5):uniform()
    target[1]:fill(1)
    local err1 = criterion:forward(input,target)
    
    -- fill index with the ignore label
    target[1]:fill(0)-- force this value to at least be 5
    local err2 = criterion:forward(input,target)
    
    -- fill index with a different value than the ignore label
    target[1]:fill(2)-- force this value to at least be 5
    local err3 = criterion:forward(input,target)
    
    tester:ne(err1,err2, precision, "err1 should be smaller to err2")
    tester:eq(err1*10,err2*9, precision, "err1 should be equal to err2")
    tester:ne(err1,err3, precision, "err1 should be different to err3")
end

function mytest.SingleFwdCrossEntropyCriterionIgnoreTensorLabel()
-- Do a forward pass through MSECriterion and it should give different results 

    -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, torch.Tensor{1,2,3,4,5}) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    input[1]:fill(1)
    local target = torch.Tensor(10,5):uniform()
    target[1]:fill(1)
    local err1 = criterion:forward(input,target)
    
    -- fill index with the ignore label
    target[1] = torch.Tensor({1,2,3,4,5})-- force this value to at least be 5
    local err2 = criterion:forward(input,target)
    
    tester:eq(err1*10,err2*9, precision, "err1 should be equal to err2")
end

function mytest.SingleFwdMSECriterionIgnoreMultipleLabels()
-- Do a forward pass through MSECriterion and it should give different results for each label

    -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, {1,2,3,4,5}) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform(); -- -torch.range(1,10):reshape(10,1):repeatTensor(1,5)
    input[1]:fill(0)
    local target = torch.Tensor(10,5):uniform()
    target[1]:fill(0)
    local err1 = criterion:forward(input,target)
    
    tester:assertgt(err1, 0, "err1 should be greater than 0")
    
    -- fill index with the ignore label
    for _, v in pairs({1,2,3,4,5}) do
        target[1]:fill(v)
        local err2 = criterion:forward(input,target)
        tester:eq(err1*10,err2*9, precision, "err1 should be equal to err2")
    end
end

function mytest.SingleFwdMSECriterionIgnoreLabelALL()
-- Do a forward pass through MSECriterion and it should give 0.

     -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, 1) -- set to ignore class 0

    local input = torch.Tensor(10,5):fill(1)
    local target = torch.Tensor(10,5):fill(1)
    local err1 = criterion:forward(input,target)
    
    tester:eq(err1, 0, "err1 should be equal to 0")
end

function mytest.SingleFwdBwdMSECriterionTensorTypes()
-- Do a forward pass through ClassNLLCriterion and it should give the same error even if the tensor is shuffled.

    -- define criterion
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Single(mse, 1) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10,5):uniform()
    local err1 = criterion:forward(input,target)
    
    local tensorTypesTable = {
        'torch.FloatTensor',
        'torch.DoubleTensor',
        'torch.CudaTensor'
    }
    
    for i, ttype in pairs(tensorTypesTable) do
        if ttype == 'torch.CudaTensor' then
            require 'cutorch'
            require 'cunn'
        end
        local inp = input:type(ttype)
        local tgt = target:type(ttype)
        criterion:type(ttype)
        local grad = criterion:backward(inp, tgt)
        tester:assertTensorNe(grad, grad:clone():fill(0), "Backward: gradient should be different than 0")
        tester:assert(grad:type() == inp:type(), "Tensor types should be equal: " .. grad:type() .. '~='..inp:type())
    end
end

function mytest.SingleFwdCrossEntropyCriterionIgnoreLabel()
-- Do a forward pass through ClassNLLCriterion and it should give different results

    -- define criterion
    local nll = nn.CrossEntropyCriterion()
    local criterion = criterion_filter.Single(nll, 1) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(2,5)
    local err1 = criterion:forward(input,target)
    
    -- shuffle indexes
    target[1]=1 -- force this value to at least be 5
    local err2 = criterion:forward(input,target)

    tester:ne(err1, err2, "err1 and err2 should be different")
end

function mytest.SingleFwdCrossEntropyCriterionIgnoreTwoLabels()
-- Do a forward pass through ClassNLLCriterion and it should give different results

    -- define criterion
    local nll = nn.CrossEntropyCriterion()
    local criterion = criterion_filter.Single(nll, {4,5}) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(1,3)
    local err1 = criterion:forward(input,target)
    
    target[1]=4 -- force this value to at least be 5
    local err2 = criterion:forward(input,target)

    tester:ne(err1, err2, "err1 and err2 should be different")
    
    target[1]=5 -- force this value to at least be 5
    local err3 = criterion:forward(input,target)

    tester:ne(err1, err3, "err1 and err2 should be different")
    tester:eq(err2, err3, precision, "err1 and err2 should be equal")
    
    target[1]=5 -- force this value to at least be 5
    target[2]=4 -- force this value to at least be 5
    local err4 = criterion:forward(input,target)
    
    tester:ne(err2, err4, "err2 and err4 should be different")
    tester:ne(err3, err4, "err3 and err4 should be different")
end


function mytest.SingleFwdCrossEntropyCriterionIgnoreLabelsRandom()
-- Do a forward pass through CrossEntropyCriterion over many different tensor sizes and labels

    -- define criterion
    local nll = nn.CrossEntropyCriterion()
    
    local data_max_size = 32
    local data_max_classes = 32
    local max_num_iters = 10
    local max_num_classes_ignore = data_max_classes/2
    
    local function getRandomIgnoreLabels(nClasses, nIgnore)
        local rand = torch.randperm(nClasses)
        local max_samples = math.min(nClasses, nIgnore)
        return rand:index(1,torch.range(1,max_samples):long()):totable()
    end
    
    local function getNonIgnoredLabel(nClasses,ignoreLabels)
        while true do
            local flag_good = true
            local label = torch.random(1,nClasses)
            for i=1, #ignoreLabels do
                if label == ignoreLabels[i] then
                    flag_good = false
                end
            end
            if flag_good then
                return label
            end
        end        
    end
    
    
    for i=1, max_num_iters do
      
        local size_data = torch.random(10,data_max_size)
        local n_classes = torch.random(1,data_max_classes)
        local n_ignore_labels = torch.random(1,math.min(n_classes,max_num_classes_ignore))
        local ignore_classes = getRandomIgnoreLabels(n_classes, n_ignore_labels)
        local good_label = getNonIgnoredLabel(n_classes, ignore_classes)
        local n_samples = math.floor(torch.random(1,size_data)/2)
        
        local input = torch.Tensor(size_data,n_classes):uniform()
        local target = torch.Tensor(size_data):random(1,n_classes)
        target:indexFill(1, torch.range(1,n_samples):long(), good_label)
        
        local criterion = criterion_filter.Single(nll, ignore_classes) -- set to ignore class 0

        local err1 = criterion:forward(input,target)
        
        -- populate data with ignore labels
        local ind = torch.randperm(size_data)
        for i=1, n_samples do
            local random_label = torch.random(1,n_ignore_labels)
            target[i] = ignore_classes[random_label]
        end
        local err2 = criterion:forward(input,target)
        tester:ne(err1, err2, "err1 and err2 should be different (iter"..i..')')
    end
    
end

--===================================================
-- Parallel Criterion
--===================================================

function mytest.PrlFwdBwdSingleClassNLLCriterionNoIgnoreLabel()

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local criterion = criterion_filter.Parallel():add(nll, 1, 1) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(1,5)
    local err1 = criterion:forward({input},{target})
    
    -- shuffle indexes
    local shuffledIdx = torch.randperm(10):long()
    local err2 = criterion:forward({input:index(1,shuffledIdx)},{target:index(1,shuffledIdx)})

    tester:eq(err1, err2, "err1 and err2 should be equal")
    
    local grad = criterion:backward({input}, {target})
    
    tester:assertTableNe(grad, {input:clone():fill(0)}, "Backward: gradient should be different than 0")
end

function mytest.PrlFwdBwdMultipleClassNLLCriterionNoIgnoreLabel()

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local criterion = criterion_filter.Parallel()
    criterion:add(nll, 1, 1) -- set to ignore class 0
    criterion:add(nll, 1, 2) -- set to ignore class 0
    criterion:add(nll, 1, 3) -- set to ignore class 0
    criterion:add(nll, 1, 4) -- set to ignore class 0
    criterion:add(nll, 1, 5) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(1,5)
    local err1 = criterion:forward({input,input,input,input,input},{target,target,target,target,target})
    
    -- shuffle indexes
    local shuffledIdx = torch.randperm(10):long()
    input = input:index(1,shuffledIdx)
    target = target:index(1,shuffledIdx)
    local err2 = criterion:forward({input,input,input,input,input},{target,target,target,target,target})

    tester:eq(err1, err2, "err1 and err2 should be equal")
end

function mytest.PrlFwdBwdMultipleCriterionsIgnoreLabels()

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Parallel()
    criterion:add(nll, 1, {1,4}) -- set to ignore class 0
    criterion:add(mse, 1, {0, torch.Tensor{1,1,1,1,1}}) -- set to ignore class 0

    local input = {torch.Tensor(10,5):uniform(), torch.Tensor(10,5):uniform()}
    local target = {torch.Tensor(10):random(1,5), torch.Tensor(10,5):uniform()}
    local err1 = criterion:forward(input, target)
    
    tester:ne(err1, 0, precision, "err1 should be equal to 0")
end


function mytest.PrlFwdBwdMultipleCriterionsIgnoreLabelsALL()

    -- define criterion
    local nll = nn.ClassNLLCriterion()
    local mse = nn.MSECriterion()
    local criterion = criterion_filter.Parallel()
    criterion:add(nll, 1, 1) -- set to ignore class 0
    criterion:add(mse, 1, 1) -- set to ignore class 0

    local input = {torch.Tensor(10,5):fill(1), torch.Tensor(10,5):fill(1)}
    local target = {torch.Tensor(10):fill(1), torch.Tensor(10,5):fill(1)}
    local err1 = criterion:forward(input, target)
    
    tester:eq(err1, 0, precision, "err1 should be equal to 0")
    
    local grad = criterion:backward(input, target)
    
    tester:assertTableEq(grad, {torch.Tensor(10,5):fill(0), torch.Tensor(10,5):fill(0)}, precision, "Backward: gradient should be equal to 0")
end


function mytest.PrlFwdBwdSingleCrossEntropyCriterionNoIgnoreLabel()

    -- define criterion
    local nll = nn.CrossEntropyCriterion()
    local criterion = criterion_filter.Parallel():add(nll, 1, 1) -- set to ignore class 0

    local input = torch.Tensor(10,5):uniform()
    local target = torch.Tensor(10):random(1,5)
    local err1 = criterion:forward({input},{target})
    
    -- shuffle indexes
    local shuffledIdx = torch.randperm(10):long()
    local err2 = criterion:forward({input:index(1,shuffledIdx)},{target:index(1,shuffledIdx)})

    tester:eq(err1, err2, precision, "err1 and err2 should be equal")
    
    local grad = criterion:backward({input}, {target})
    
    tester:assertTableNe(grad, {input:clone():fill(0)}, "Backward: gradient should be different than 0")
end

--===================================================
-- Examples
--===================================================

function mytest.SingleCriterionExample1()
--[[ Example using `ClassNLLCriterion`. 
Computes the loss for an input tensor with all allowed targets,
and proceed to compute the loss with a blacklisted label. 
]]

    -- 1. define model
    model = nn.Sequential()
    model:add(nn.Linear(10,4))

    -- 2. define criterion
    nll = nn.ClassNLLCriterion()
    criterion = criterion_filter.Single(nll, 4) -- set to ignore class 0

    -- 3. define input data 
    input = torch.Tensor(5,10):uniform()
    target = torch.Tensor(5):random(1,3)

    -- 4. compute loss (forward pass only)
    -- 4.1. (no labels to be ignored at this point)
    output = model:forward(input)
    err1 = criterion:forward(output,target)

    -- 4.2. set one target label to 0 (to be ignored)
    target[5] = 4
    err2 = criterion:forward(output,target)
    tester:ne(err1,err2,'err1 and err2 should be different')
end

function mytest.SingleCriterionExample2()
--[[ Example using `MSECriterion`.
Computes the loss and gradient for an input tensor + targets,
and proceeds to compute the loss and gradient with blacklisted labels.
]]

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
    gradOutput1 = criterion:backward(output,target):clone()

    -- 4.2. set one target to 0's (to be ignored)
    target[1]:fill(0)
    err2 = criterion:forward(output,target)
    gradOutput2 = criterion:backward(output,target):clone()

    --4.3. set another target to 0's and one to {1,2,3,4} 
    target[3]:fill(0)
    target[4]:copy(torch.Tensor{1,2,3,4})
    err3 = criterion:forward(output,target)
    gradOutput3 = criterion:backward(output,target):clone()
    
    tester:ne(err1,err2,'err1 and err2 should be different')
    tester:ne(err2,err3,'err2 and err3 should be different')
    
    tester:ne(gradOutput1,gradOutput2,'gradOutput1 and gradOutput2 should be different')
    tester:ne(gradOutput1,gradOutput3,'gradOutput1 and gradOutput3 should be different')
    tester:ne(gradOutput2,gradOutput3,'gradOutput2 and gradOutput3 should be different')
end

function mytest.ParallelCriterionExample1()
--[[ Example using multiple criterions (`ClassNLLCriterion` + `ClassNLLCriterion`).
Here, the loss is computed using two equal criterias with equal weigths. 
Check how the loss value differs when it is computed with ignored/filtered/blacklisted labels.
]]
    -- 1. define model
    model = nn.Sequential()
    prl = nn.ConcatTable()
    prl:add(nn.Linear(10,10))
    prl:add(nn.Linear(10,10))
    model:add(prl)

    -- 2. define criterions
    criterion = criterion_filter.Parallel()
    criterion:add(nn.ClassNLLCriterion(), 1, 6) -- set different ignore labels
    criterion:add(nn.ClassNLLCriterion(), 1, 7) -- set different ignore labels

    -- 3. define input data
    input = torch.Tensor(5,10):uniform()
    target1 = torch.Tensor(5):random(1,10)
    target1[1]=1
    target2 = torch.Tensor(5):random(1,10)
    target2[1]=1

    -- 4. compute loss
    -- 4.1. without ignore labels
    output = model:forward(input)
    err1 = criterion:forward(output,{target1, target2})

    -- 4.2. now with ignored/blacklisted labels
    target1[1] = 6
    target2[1] = 7 
    err2 = criterion:forward(output,{target1, target2})

    -- 4.3. flip labels 
    target1[1] = 7
    target2[1] = 6 
    err3 = criterion:forward(output,{target1, target2})
    
    tester:ne(err1,err2,'err1 and err2 should be different')
    tester:ne(err1,err3,'err1 and err2 should be different')
end

function mytest.ParallelCriterionExample2()
--[[ Example using multiple criterions (`ClassNLLCriterion` + `MSECriterion`).
Here, the loss is computed using two different criterias and weigths. 
Check how the loss value differs when it is computed with ignored/filtered/blacklisted labels.
]]

    -- 1. define model
    model = nn.Sequential()
    prl = nn.ConcatTable()
    prl:add(nn.Linear(10,5))
    prl:add(nn.Linear(10,4))
    model:add(prl)

    -- 2. define criterions
    criterion = criterion_filter.Parallel()
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

    -- 4.2. now with ignored/blacklisted labels
    target2[2]:fill(1)
    target2[5]:fill(1)
    err2 = criterion:forward(output, {target1, target2})
    
    tester:ne(err1,err2,'err1 and err2 should be different')
end

-- Run tests
tester:add(mytest)
tester:run()
