![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

## 1.Project Overview

We provided an introduction to the main functions, including the fractional-order program, the FFCM model function, and the Runge-Kutta solver. The details are as follows:

- `/controldiffeq/cdeint_module.py` contains the main function, which is primarily responsible for implementing the FFCM model.
- `/lib/gamma.py` contains the G-L fractional order function.
- `/torchdiffeq/_impl/rk_common.py`contains the Runge-Kutta solver.


## 2.Code Details

<p align="center">
<img src=".\pic\FFCM.png" height = "360"   alt="" align=center />  
<br><br>
</p>

| Formula Variable | Program Variable Name | 🔎  Line Range in Code |
| --- | --- | ---|
|$$g[a(t_b)]$$| self.A_0_NEW  |cdeint_module.py/lines 44-52  |
| $$g[a(t_1)]$$ | self.A_N_OLD | cdeint_module.py/lines 33-40 |
| fractional order $$\\alpha $$  | self.fract  | cdeint_module.py/line 24 |
| coefficient $$l$$      | self.l  |cdeint_module.py/line 19 |
| gain coefficient $$\\lambda $$ | self.lambd| cdeint_module.py/line 18|
| $$M$$ | constant  |cdeint_module.py/line 105|
| $$N_k$$|   gamma.gamma_all| gamma.py/lines 126-142|
|Third-order Runge-Kutta    |  rk3_alt_step_func|rk_common.py/ lines 103-112|
| Fourth-order Runge-Kutta | rk4_alt_step_func |rk_common.py/ lines 114-123|
| Fifth-order Runge-Kutta | rk5_alt_step_func |rk_common.py/ lines 125-135|



## 3.Requirements

- Python 3.9
- numpy ==  1.26.3
- pandas == 2.2.3
- torch ==  2.6.0
- tqdm == 4.67.1




