--[[
    Description:
        Filters the grad output tensor values matching an input label(s). If a tensor has the same value as the label, the i'th position in the gradInput var if filled with 0's.
]]

require('torch')
require('nn')
local ParallelCriterionFilterLabel, parent = torch.class('criterion_filter.Parallel', 'nn.Criterion')

function ParallelCriterionFilterLabel:__init(repeatTarget)
  parent.__init(self)
  self.target_double = torch.DoubleTensor() -- this var is used to do tensor matching only!
  self.criterions = {}
  self.weights = {}
  self.gradInput = {}
  self.filterLabel = {}
  self.repeatTarget = repeatTarget
end


function ParallelCriterionFilterLabel:getFilteredIndexes(target, filterLabel, flag)
  local indexes = {}
  if not next(filterLabel) then
    -- filter label table is empty, return all indexes
    for i=1, target:size(1) do table.insert(indexes,i) end
    return indexes
  end
  
  self.target_double:resize(target:size()):copy(target) -- WARNING: type is cast to double for tensor matching purposes only!
  if flag == 0 then
      -- fetch indexes to NOT be filtered/ignored
    if target:dim() > 1 then
      for i=1, self.target_double:size(1) do
        for j=1, #filterLabel do
          if not (torch.add(self.target_double[i],-filterLabel[j]):sum() == 0) then
            indexes[i] = 1
          end      
        end
      end
    else
      for i=1, self.target_double:size(1) do
        for j=1, #filterLabel do
          if not (self.target_double[i] == filterLabel[j]) then 
            indexes[i] = 1
          end      
        end
      end
    end
  else
    -- fetch indexes to be filtered/ignored
    if target:dim() > 1 then
      -- compare tensors
      for i=1, self.target_double:size(1) do
        for j=1, #filterLabel do
          if torch.add(self.target_double[i],-filterLabel[j]):sum() == 0 then
            indexes[i] = 1
          end      
        end
      end
    else
      --compare numbers
      for i=1, self.target_double:size(1) do
        for j=1, #filterLabel do
          if not (self.target_double[i] == filterLabel[j]) then 
            indexes[i] = 1
          end      
        end
      end
    end
  end
  -- convert hash indexes to table entries
  local indexTable = {}
  for k, _ in pairs(indexes) do table.insert(indexTable, k) end  
  return indexTable
end


function ParallelCriterionFilterLabel:add(criterion, weight, ignore)
  -- Ignore/filter label can be either a single value (number) or tensor, or multiple values (table of numbers/tensors). 
  if ignore then
    if not (type(ignore) == 'number' or type(ignore) == 'userdata' or type(ignore) == 'table') then
      assert(false, 'Ignore/filter label must be either a number or a Tensor. Current type is: ' .. type(ignore))
    end    
  end
  assert(criterion, 'no criterion provided')
  weight = weight or 1
  table.insert(self.criterions, criterion)
  table.insert(self.weights, weight)
  table.insert(self.filterLabel, self:setIgnoreLabels(ignore))
  return self
end


function ParallelCriterionFilterLabel:setIgnoreLabels(ignore_label)
  local ignore_table = {}
  if ignore_label then
    -- add ignore/filter labels
    if type(ignore_label) == 'table' then
      for i=1, #ignore_label do
        assert((type(ignore_label[i]) == 'number' or type(ignore_label[i]) == 'userdata' or type(ignore_label[i]) == 'table'), 'ignore[i] must be a number, table or Tensor.')
        -- check if the label is a number (all number types are, by default, converted to tensors)
        if type(ignore_label[i]) == 'number' then
          table.insert(ignore_table, ignore_label[i])
        elseif type(ignore_label[i]) == 'table' then
          table.insert(ignore_table, torch.DoubleTensor(ignore_label[i]))
        else
          table.insert(ignore_table, ignore_label[i]:double())
        end
      end
    else
      -- check if the label is a number (all number types are, by default, converted to tensors)
      if type(ignore_label) == 'number' then
        table.insert(ignore_table, ignore_label)
      else
        table.insert(ignore_table, ignore_label:double())
      end
    end
  end
  return ignore_table
end


function ParallelCriterionFilterLabel:updateOutput(input, target)
  self.output = 0
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    local filterLabel = self.filterLabel[i]
    -- find indexes to not be ignored (if any)
    local input_filtered, target_filtered
    if next(self.filterLabel) then
      local indexes = self:getFilteredIndexes(target, filterLabel, 0) --fetch indexes to compute the loss
      if next(indexes) then
        input_filtered = input[i]:index(1,torch.LongTensor(indexes))
        target_filtered = target:index(1,torch.LongTensor(indexes))
      else
        -- empty table, set some temporary tensors
        input_filtered, target_filtered = torch.Tensor({0}), torch.Tensor({0})
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
      if next(indexes) then criterion_gradInput:indexFill(1,torch.LongTensor(indexes),0) end
    end
    nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion_gradInput)
  end
  return self.gradInput
end


function ParallelCriterionFilterLabel:type(type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end