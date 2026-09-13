# Architecture

## Generic layer

`neuralcert.core` defines only contracts and value objects. A problem plugin
implements domain sampling, a differentiable objective, and inexpensive
validation. Algorithms live in separate stages:

```text
Problem
  │
  ▼
Discovery ──> Distillation ──> Refinement ──> Verification
    │              │               │                │
 numerical       structural      local/exact      independent
```

The pipeline preserves every intermediate output as a typed stage result. No
stage may silently promote a numerical result to a proof.

`neuralcert.problems.maynard` is a plugin and adapter layer. The specialized
historical implementation remains under `maynard_tools` for now, so the new
abstraction does not rewrite proven numerical code without separate regression
validation.

## Hard boundary

`maynard_tools.discovery` must not import modules from
`maynard_tools.certification`. An AST test enforces this rule, including dynamic
imports whose module is supplied as a string literal.

The reason is substantive: discovery searches for candidates quickly and
numerically, while certification must independently establish exactly what has
been proved.

## Dependency direction

```text
maynard_tools.discovery ──> NumPy / SciPy / PyTorch
maynard_tools.certification ──> NumPy / SciPy / optional FLINT
```

Certification may read discovery NPZ output as **data**. This is not a Python
import and creates no code dependency in the opposite direction.

## Historical variants

The two discovery entry points remain separate because the gated variant has
its own distributed execution model and more extensive objective logic. Only
definitions shown to be identical have been moved into shared modules.

The certification variants remain explicit backends. This allows an existing
result to be checked again with the same method, without silently switching to
a different numerical method.

The public discovery and certification CLIs use the same method names: `poly`
for the existing polynomial/neural workflow and `ratio` for the confluent-ratio
workflow. The ratio certifier uses Arb with directed rounding and currently
accepts only one power-1 channel.

## Independent verifier

`maynard_tools.verifier` is included in the same wheel only for ease of
distribution. Its implementation imports nothing from `discovery` or
`certification` and has no relative imports into other package code. An AST test
enforces both rules. The certificate verifier uses only the standard library
for the poly route and loads NumPy and python-flint only inside the ratio route.
The direct I/J check uses NumPy and is explicitly non-rigorous.
