require('torch')
require('nn')

criterion_filter = {}

-- Add functions
-- todo: Extend to single criterions
require('criterion_filter.Parallel')
require('criterion_filter.Single')

return criterion_filter