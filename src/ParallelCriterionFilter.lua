--[[
    Description:
        Filters the grad output tensor values matching an input label(s). If a tensor has the same value as the label, the i'th position in the gradInput var if filled with 0's.
]]

local ParallelCriterionFilterLabel, parent = torch.class('criterion_filter.Parallel', 'nn.Criterion')

function ParallelCriterionFilterLabel:__init(repeatTarget)
    parent.__init(self)
    self.criterions = {}
    self.weights = {}
    self.gradInput = {}
    self.filterLabel = {}
    self.repeatTarget = repeatTarget
end

-- add ignore/filter labels
function ParallelCriterionFilterLabel:setIgnoreLabels(ignore_label)
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
                error('Table values type undefined: ' .. type(v)..'. Values must be either table, number or tensor types.')
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

function ParallelCriterionFilterLabel:getFilteredIndexes(target, filterLabel, flag)
    local indexes = {}
    if flag == 0 then
        -- fetch indexes to NOT be filtered/ignored
        if target:dim() > 1 then
            for k, v in pairs(filterLabel) do
                for i=1, target:size(1) do
                    if not (torch.add(target[i],-v):sum() == 0) then
                        indexes[i] = 1
                    end      
                end
            end
        else
            for k, v in pairs(filterLabel) do
                for i=1, target:size(1) do
                    if not (target[i] == v) then 
                        indexes[i] = 1
                    end      
                end
            end
        end
    else
        -- fetch indexes to be filtered/ignored
        if target:dim() > 1 then
            -- compare tensors
            for k, v in pairs(filterLabel) do
                for i=1, target:size(1) do
                    if torch.add(target[i],-v):sum() == 0 then
                        indexes[i] = 1
                    end      
                end
            end
        else
            --compare numbers
            for k, v in pairs(filterLabel) do
                for i=1, target:size(1) do
                    if not (target[i] == v) then 
                        indexes[i] = 1
                    end
                end
            end
        end
    end
    -- convert hash indexes to table entries
    local indexTable = {}
    for k, _ in pairs(indexes) do table.insert(indexTable, k) end  
    return torch.LongTensor(indexTable)
end

-- Ignore/filter label can be either a single value (number) or tensor, or multiple values (table of numbers/tensors). 
function ParallelCriterionFilterLabel:add(criterion, weight, ignore)
    if ignore then
        if not (type(ignore) == 'number' or type(ignore) == 'userdata' or type(ignore) == 'table') then
            error('Ignore/filter label must be either a number or a Tensor. Current type is: ' .. type(ignore))
        end
    end
    assert(criterion, 'no criterion provided')
    weight = weight or 1
    table.insert(self.criterions, criterion)
    table.insert(self.weights, weight)
    table.insert(self.filterLabel, self:setIgnoreLabels(ignore))
    return self
end

function ParallelCriterionFilterLabel:updateOutput(input, target)
    self.output = 0
    for i,criterion in ipairs(self.criterions) do
        local target = self.repeatTarget and target or target[i]
        local filterLabel = self.filterLabel[i]
        -- find indexes to not be ignored (if any)
        local input_filtered, target_filtered
        if next(self.filterLabel) then
            --fetch indexes to compute the loss
            local indexes = self:getFilteredIndexes(target, filterLabel, 0) 
            if indexes:numel()>0 then
                input_filtered = input[i]:index(1,indexes)
                target_filtered = target:index(1,indexes)
            else
                -- empty table, set some temporary tensors
                input_filtered = torch.Tensor({0}):typeAs(input)
                target_filtered = input_filtered:clone()
            end
        else
            input_filtered, target_filtered = input[i], target
        end
        self.output = self.output + self.weights[i]*criterion:updateOutput(input_filtered, target_filtered)
    end
    return self.output
end

function ParallelCriterionFilterLabel:updateGradInput(input, target)
    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
    nn.utils.recursiveFill(self.gradInput, 0)
    for i,criterion in ipairs(self.criterions) do
        local target = self.repeatTarget and target or target[i]
        local criterion_gradInput = criterion:updateGradInput(input[i], target)
        if next(self.filterLabel) then
            local indexes = self:getFilteredIndexes(target, 1) 
            if indexes:numel()>0 then criterion_gradInput:indexFill(1,torch.indexes,0) end
        end
        nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion_gradInput)
    end
    return self.gradInput
end

function ParallelCriterionFilterLabel:type(type, tensorCache)
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end

function ParallelCriterionFilterLabel:cuda()
    for i=1, #self.filterLabel do
        if string.match(torch.type(self.filterLabel[i]),'Tensor') and string.match(torch.type(self.filterLabel[i]),'torch') then
            self.filterLabel[i] = self.filterLabel[i]:cuda()
        end
    end
    return self:type('torch.CudaTensor')
end