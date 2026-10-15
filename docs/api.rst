API reference
=============

Scheduler
---------

.. autoclass:: atombatch.Scheduler
   :members: add_model, submit, run

   .. attribute:: batch_log

      One entry ``(model, structures, atoms)`` for every batched call made by
      :meth:`run`.

.. autoclass:: atombatch.TaskContext
   :members: calculator, evaluate, linked_calculators

.. autoclass:: atombatch.TaskResult
   :members:

.. autoclass:: atombatch.BatchCalculator
   :members:
   :exclude-members: calculate, implemented_properties

Tasks
-----

.. autofunction:: atombatch.single_point
.. autofunction:: atombatch.relax
.. autofunction:: atombatch.neb
.. autofunction:: atombatch.molecular_dynamics

Workflows
---------

.. autofunction:: atombatch.displacement_forces
.. autofunction:: atombatch.equation_of_state
.. autofunction:: atombatch.swap_monte_carlo

Random structure searching
--------------------------

.. autofunction:: atombatch.airss
.. autofunction:: atombatch.hot_airss
.. autofunction:: atombatch.workflows.airss.random_structure
.. autofunction:: atombatch.workflows.airss.buildcell
.. autofunction:: atombatch.workflows.airss.rank

Backends
--------

.. autoclass:: atombatch.Backend
   :members: evaluate

.. autoclass:: atombatch.SymmetrixBackend
.. autoclass:: atombatch.TorchSimBackend
.. autoclass:: atombatch.FunctionBackend
.. autoclass:: atombatch.SequentialBackend

Symmetrix graph batching
------------------------

.. automodule:: atombatch.backends.symmetrix
   :members: GraphBatcher, supports_graph_batching, edge_forces, segment_virials

atomate2
--------

.. autofunction:: atombatch.integrations.atomate2.atomate2_task
