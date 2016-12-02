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

function SingleCriterionFilterLabel:getFilteredIndexes(target, filterLabel, flag)
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

function SingleCriterionFilterLabel:updateOutput(input, target)
    self.output = 0
    assert(torch.type(input) == torch.type(target), ('Target and input type mismatch: %s ~= %s'):format(torch.type(input), torch.type(target)))
    local input_filtered, target_filtered
    if next(self.filterLabel) then
        local indexes = self:getFilteredIndexes(target, self.filterLabel, 0) --fetch indexes to compute the loss
        if indexes:numel()>0 then
            input_filtered = input:index(1,indexes)
            target_filtered = target:index(1,indexes)
        else
            -- empty table, set some temporary tensors
            input_filtered = torch.Tensor({0}):typeAs(input)
            target_filtered = input_filtered:clone()
        end
    else
        input_filtered, target_filtered = input, target
    end
    self.output = self.criterion:updateOutput(input_filtered, target_filtered)
    return self.output
end

function SingleCriterionFilterLabel:updateGradInput(input, target)
    local criterion_gradInput = self.criterion:updateGradInput(input, target)
    if next(self.filterLabel) then
        local indexes = self:getFilteredIndexes(target, self.filterLabel, 1) 
        if indexes:numel()>0 then criterion_gradInput:indexFill(1,indexes,0) end
    end
    self.gradInput = criterion_gradInput
    return self.gradInput
end

function SingleCriterionFilterLabel:cuda()
    for i=1, #self.filterLabel do
        if string.match(torch.type(self.filterLabel[i]),'Tensor') and string.match(torch.type(self.filterLabel[i]),'torch') then
            self.filterLabel[i] = self.filterLabel[i]:cuda()
        end
    end
    return self:type('torch.CudaTensor')
end