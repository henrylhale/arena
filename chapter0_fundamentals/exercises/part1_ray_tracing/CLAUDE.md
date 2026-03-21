# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **ARENA 3.0 Chapter 0, Part 1: Ray Tracing** — a series of exercises implementing a ray tracer from scratch using PyTorch tensor operations. The exercises progress from 1D ray-segment intersection to full 3D mesh raytracing with Lambert shading.

## Running Tests

Run all pytest tests (comparing `answers.py` against `solutions.py`):
```bash
pytest test_with_pytest.py
```

Run a single test function:
```bash
pytest test_with_pytest.py::test_intersect_ray_1d
pytest test_with_pytest.py::test_intersect_rays_1d
pytest test_with_pytest.py::test_triangle_ray_intersects
```

Tests in `tests.py` are called inline inside `answers.py` and `solutions.py` (e.g., `tests.test_intersect_ray_1d(intersect_ray_1d)`), so running `python answers.py` also runs tests.

## File Structure and Architecture

- **`answers.py`** — Where the user writes their implementations. Imports from `tests.py` and `utils.py`, and calls tests inline after each function.
- **`solutions.py`** — Reference implementations. All top-level code is guarded by `if MAIN:` so it only runs when executed directly. Also defines shared data (`rays1d`, `segments`) that `tests.py` imports.
- **`tests.py`** — Test functions that take a user function as argument and compare it against `solutions.py`. Called inline in `answers.py` and also used by `test_with_pytest.py`.
- **`test_with_pytest.py`** — Pytest wrappers that import from `answers` module and compare against `solutions`. Skips gracefully if `answers` module doesn't exist.
- **`utils.py`** — Plotly visualization helpers: `render_lines_with_plotly`, `setup_widget_fig_ray`, `setup_widget_fig_triangle`. Uses `renderer="browser"` for display.

## Path Setup

The parent `exercises/` directory is added to `sys.path` so that imports work as:
```python
import part1_ray_tracing.solutions as solutions
import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import render_lines_with_plotly
from plotly_utils import imshow  # lives in exercises/ dir
```

This means `pytest` must be run from within `part1_ray_tracing/` or the `exercises/` directory must be on the path.

## Key Conventions

- **Tensor shapes** are documented with `jaxtyping` annotations: `Float[Tensor, "nrays 2 3"]` means a float tensor of shape `(nrays, 2, 3)`.
- **Ray representation**: shape `(2, 3)` where `[0]` is origin `O` and `[1]` is direction `D`.
- **Segment representation**: shape `(2, 3)` where `[0]` is `L_1` and `[1]` is `L_2`.
- **Triangle representation**: shape `(3, 3)` for vertices `A`, `B`, `C`.
- Singular matrices (parallel rays/segments) are handled by detecting `det(A).abs() < 1e-8` and replacing with the identity matrix, then masking those results out.
- GPU functions use `device = "cuda"` and return results on CPU via `.cpu()`.

## Dependencies

PyTorch (`torch`), `einops`, `jaxtyping`, `plotly`, `ipywidgets`, `tqdm`.
