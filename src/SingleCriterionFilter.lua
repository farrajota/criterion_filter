--[[
    Description:
        Filters the grad output tensor values matching an input label(s). If a tensor has the same value as the label, the i'th position in the gradInput var if filled with 0's.
]]

require('torch')
require('nn')
local SingleCriterionFilterLabel, parent = torch.class('criterion_filter.Single', 'nn.Criterion')

function SingleCriterionFilterLabel:__init(criterion, ignore_label)
  parent.__init(self)
  assert(criterion, 'Must insert a criterion for evaluation.')
  if ignore_label then
    if not (type(ignore_label) == 'number' or type(ignore_label) == 'userdata' or type(ignore_label) == 'table') then
      assert(false, 'Ignore/filter label must be either a number or a Tensor. Current type is: ' .. type(ignore_label))
    end    
  end
  self.criterion = criterion
  self.target_double = torch.DoubleTensor() -- this var is used to do tensor matching only!
  self.filterLabel = self:setIgnoreLabels(ignore_label)
end


function SingleCriterionFilterLabel:setIgnoreLabels(ignore_label)
  if ignore_label then
    if not (type(ignore_label) == 'number' or type(ignore_label) == 'userdata' or type(ignore_label) == 'table') then
      assert(false, 'Ignore/filter label must be either a number or a Tensor. Current type is: ' .. type(ignore_label))
    end    
  end
  local ignore_table = {}
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
  return ignore_table
end


function SingleCriterionFilterLabel:getFilteredIndexes(target, flag)
  local indexes = {}
  self.target_double:resize(target:size()):copy(target) -- WARNING: type is cast to double for tensor matching purposes only!
  if flag == 0 then
      -- fetch indexes to NOT be filtered/ignored
    if target:dim() > 1 then
      for i=1, self.target_double:size(1) do
        for j=1, #self.filterLabel do
          if not (torch.add(self.target_double[i],-self.filterLabel[j]):sum() == 0) then
            indexes[i] = 1
          end      
        end
      end
    else
      for i=1, self.target_double:size(1) do
        for j=1, #self.filterLabel do
          if not (self.target_double[i] == self.filterLabel[j]) then 
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
        for j=1, #self.filterLabel do
          if torch.add(self.target_double[i],-self.filterLabel[j]):sum() == 0 then
            indexes[i] = 1
          end      
        end
      end
    else
      --compare numbers
      for i=1, self.target_double:size(1) do
        for j=1, #self.filterLabel do
          if not (self.target_double[i] == self.filterLabel[j]) then 
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


function SingleCriterionFilterLabel:updateOutput(input, target)
  self.output = 0
  local input_filtered, target_filtered
  if next(self.filterLabel) then
    local indexes = self:getFilteredIndexes(target, 0) --fetch indexes to compute the loss
    if next(indexes) then
      input_filtered = input:index(1,torch.LongTensor(indexes))
      target_filtered = target:index(1,torch.LongTensor(indexes))
    else
      -- empty table, set some temporary tensors
      input_filtered, target_filtered = torch.Tensor({0}), torch.Tensor({0})
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
    local indexes = self:getFilteredIndexes(target, 1) 
    if next(indexes) then criterion_gradInput:indexFill(1,torch.LongTensor(indexes),0) end
  end
  self.gradInput = criterion_gradInput
  return self.gradInput
end
