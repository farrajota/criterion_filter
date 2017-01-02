--[[
    Description:
        Filters the grad output tensor values matching an input label(s). If a tensor has the same value as the label, the i'th position in the gradInput var if filled with 0's.
]]

local SingleCriterionFilterLabel, parent = torch.class('criterion_filter.Single', 'nn.Criterion')

function SingleCriterionFilterLabel:__init(criterion, ignore_label)
    assert(criterion, 'Must insert a criterion for evaluation.')
    if ignore_label then
        if not (type(ignore_label) == 'number' or type(ignore_label) == 'userdata' or type(ignore_label) == 'table') then
            error('Ignore/filter label must be either a number or a Tensor. '.. 
                  'Current type is: ' .. type(ignore_label))
        end    
    end
  
    parent.__init(self)
    self.criterion = criterion
    self.filterLabel = self:setIgnoreLabels(ignore_label)
end

-- add ignore/filter labels
function SingleCriterionFilterLabel:setIgnoreLabels(ignore_label)
    local labels = {}
    if not ignore_label then
        return labels
    end
    
    if type(ignore_label) == 'table' then
        for k,v in pairs(ignore_label) do
            if type(v) == 'table' then
                table.insert(labels, torch.Tensor(v))
            elseif type(v) == 'number' then
                table.insert(labels, v)
            elseif type(v) == 'userdata' then
                table.insert(labels, v)
            else
                error('Table values type undefined: ' .. type(v).. '. Values must be either table, number or tensor types.')
            end
        end
    elseif type(ignore_label) == 'number' then
        table.insert(labels, ignore_label)
    elseif type(ignore_label) == 'userdata' then
        table.insert(labels, ignore_label)
    else
        error('ignore_label must be a number, table or Tensor.')
    end
    return labels
end

-- keep indexes to NOT be filtered/ignored
function SingleCriterionFilterLabel:keepIndexes(target, filterLabel)
    local checkEqual
    if target:dim() > 1 then
        checkEqual = function(target,v) return (torch.add(target,-v):sum() == 0) end
    else
        checkEqual = function(target,v) return (target == v) end
    end
    
    for i=1, target:size(1) do
        for k, v in pairs(filterLabel) do
            if checkEqual(target[i],v)then
                self.indexes[i]=0
                break
            end      
        end
    end
end

-- remove indexes to be filtered/ignored
function SingleCriterionFilterLabel:skipIndexes(target, filterLabel)
    local checkEqual
    if target:dim() > 1 then
        checkEqual = function(target,v) return (torch.add(target,-v):sum() == 0)  end
    else
        checkEqual = function(target,v) return (target == v) end
    end
  
    -- compare tensors
    for i=1, target:size(1) do
        for k, v in pairs(filterLabel) do
            if not checkEqual(target[i],v) then
                self.indexes[i]=0
                break
            end      
        end
    end
end

function SingleCriterionFilterLabel:getFilteredIndexes(target, filterLabel, flag)
    self.indexes = torch.range(1,target:size(1))
    if flag == 0 then
        self:keepIndexes(target, filterLabel)
    else
        self:skipIndexes(target, filterLabel)
    end
    local ind = self.indexes:gt(0):nonzero()
    if ind:numel()>0 then
        ind = ind:squeeze()
        if type(ind) == 'number' then
            ind = torch.LongTensor{ind}
        end
    end
    return ind
end

function SingleCriterionFilterLabel:updateOutput(input, target)
    self.output = 0
    local input_filtered, target_filtered = input, target
    local flag_updateOutput = true
    
    if next(self.filterLabel) then
        local indexes = self:getFilteredIndexes(target, self.filterLabel, 0) --fetch indexes to compute the loss
        if indexes:numel()>0 then
            -- Select fields from the data tensors
            input_filtered = input_filtered:index(1, indexes)
            target_filtered = target_filtered:index(1, indexes)
        else
            flag_updateOutput = false
        end
    end
    
    if flag_updateOutput then 
        self.output = self.criterion:updateOutput(input_filtered, target_filtered) 
    else
        self.output = 0
    end
    -- Some criterions (like ClassNLLCriterion) need to save the state of the forward pass. 
    -- This fixes the internal state of custom criterions that need to keep the input/target 
    -- sizes at the cost of a forward pass.
    self.criterion:updateOutput(input,target)
    return self.output
end

function SingleCriterionFilterLabel:updateGradInput(input, target)
    local criterion_gradInput = self.criterion:updateGradInput(input, target)
    if next(self.filterLabel) then
        local indexes = self:getFilteredIndexes(target, self.filterLabel, 1) 
        if indexes:numel()>0 then 
            criterion_gradInput:indexFill(1,indexes,0)
        end
    end
    self.gradInput = criterion_gradInput
    return self.gradInput
end

function SingleCriterionFilterLabel:cuda()
    for i=1, #self.filterLabel do
        if string.match(torch.type(self.filterLabel[i]),'Tensor') and 
          string.match(torch.type(self.filterLabel[i]),'torch') then
            self.filterLabel[i] = self.filterLabel[i]:cuda()
        end
    end
    return self:type('torch.CudaTensor')
end