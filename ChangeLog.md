## [0.4.02]
* hotfixes
## [0.4.01]
* bugfixes and refactoring
* smooth lambda
* evaluate well pump power in MRM
## [0.4.0]
* Add new kind of node boundary condition: PQ_Curve
* Add water choke model
* Add well PQ-curve builder (MRM module)
* Add pad PQ-curves aggregator (MRM module)
* Add well-pad-network integration modwule (MRM module)
## [0.3.24]
* A lot of bugfixes in solver io
* Bugfix fluid applying to graph
## [0.3.23]
* two crutches in dataframe io, for kind == '', and for empty altittude
## [0.3.22]
* tool for attach injection wells to net df
* fixes to run solver with injection wells
## [0.3.21]
* tool for build water net single df from two dfs
* fixes to run solver from single df water net
## [0.3.20]
* hotfix:resulted Q_m3_day on sink nodes were last cause lil bug in mixation 
## [0.3.19]
* input dataframe format changed, some columns added some removed
## [0.3.18]
* input dataframe format changed, fluid properties columns supported
## [0.3.17]
* one more heuristic to reduce iterations count in solver gradient descent
* fix critical bug with prolongation and depth mess
* saving fluid SC density to result for augmentation purpose
## [0.3.16]
* Two heuristics reduces iterations count in solver gradient descent
* Now solve() results also pushes into solver log
## [0.3.15]
23-11-2021
* Merge branch apply_predict. It used to generate neurosolver train dataset/
* Merge branch project_wells. It is a well-constructor if there is no inclinometry of well.
* Merge branch pump_replacement. It found most fit pump model to replace, if your model cannot be found in charts database
* Pump power and frequency included in results 
 Bugfixes:
* Freq == 0 means that well pump is turned off
* When solution is found, fluids should recalculated to actual P, T in attach_results
* Pump doesnt сonsume power if it is turned off

## [0.3.14]
26-10-2021
If solver.solve() cant find a solution (check solver.success flag) there are no more result attribute in graph edges and nodes objects.  

## [0.3.13]
Pump PQ curve extrapolators were sheer. It make gradient descent diverge.
Now extrapolator's А-keff is about 1e-4 and it significantly reduce solver iterations.  

## [0.3.12]
There is two changes in IO:
-It is better to give two separated dataframes to run solver, first with edges and second with nodes
-It is now user's responsibility to load auxialary data i.g. pump charts, inclinometry etc.    