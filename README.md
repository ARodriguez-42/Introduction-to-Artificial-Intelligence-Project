# README

## BnB

### Environment

```python3
pip install numpy
```

### Execution

Usage:

```
python3 BnB-submit.py <file> <heuristic_type>
```

`<heuristic_type>` can be `trivial`, `sps`, `mst`, `edge`, `mst_ed`

For example:

```python3
python3 BnB-submit.py 5_0.0_10.0.out edge
```

Get result like this:
```
Distance: 45.6082 Path: [3, 2, 4, 1, 0]
```


## SLS

### Environment

```python3
pip install numpy
```

### Execution

Usage:

```
python3 sls.py
```

For example:

```python3
python3 sls.py 

File Name: 5_0.0_10.0.out edge
Number of iterations: input desired number
```

Get result like this:
```
([Path], Distance)
Run Time 

([10, 6, 11, 7, 5, 8, 3, 4, 0, 13, 12, 2, 9, 1, 15, 14], 34.118700000000004)
0.006500244140625
```
