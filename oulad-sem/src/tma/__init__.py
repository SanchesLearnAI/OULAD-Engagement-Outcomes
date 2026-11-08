"""TMA-anchored study code (ported from oulad_analysis, see each module).

The modules here implement the main design described in
``research/TMA_TWO_WEEK_PROTOCOL_20261103.md``: anchor each module
presentation on the deadline of its first positive-weight TMA, observe
``d-13 .. d`` and ``d+1 .. d+post_days``, and compare Fail with
Pass/Distinction records whose post-deadline clicks are close.
"""
