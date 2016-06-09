package = "criterion_filter"
 version = "scm-1"
 source = {
    url = "git://github.com/farrajota/criterium_filter",
    tag = "master"
 }
 description = {
    summary = "A criterion container tailored to filter/ignore labels",
    detailed = [[
       Ignore/bypass some inputs according to a specific set of ignore labels. 
       This package allows for general, out-of-the-box criterions in torch/nn
       to be used in situations where a specific label of a certain class/output 
       is needed to be overlooked/ignored when backproping through a network model.
    ]],
    homepage = "https://github.com/farrajota/criterium_filter",
    license = "BSD",
    maintainer = "Farrajota"
 }
 dependencies = {
    "lua ~> 5.1",
    "nn >= scm-1",
    "torch >= 7.0"
 }
 build = {
  type = 'builtin',
  modules = {
      ["criterion_filter.init"] = 'init.lua',
      ["criterion_filter.Parallel"] = 'src/ParallelCriterionFilter.lua',
      ["criterion_filter.Single"] = 'src/SingleCriterionFilter.lua'
  }
 }