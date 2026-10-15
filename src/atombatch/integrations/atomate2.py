"""Run atomate2 ASE and force-field makers with batched evaluations.

atomate2 makers whose calculations go through ASE (``AseRelaxMaker`` and its
subclasses, including ``ForceFieldRelaxMaker``, ``ForceFieldStaticMaker``,
and ``EmtRelaxMaker``) obtain their calculator from a single ``calculator``
property. :func:`atomate2_task` runs such a maker's job function inside an
atombatch task with that calculator replaced by a batched one, so many
makers and structures share each model call. The task returns the document
the atomate2 job would return.

The model registered with :meth:`atombatch.Scheduler.add_model` performs the
evaluations. The force-field settings of the maker (``force_field_name``,
``calculator_kwargs``) are not used to build a calculator and only label the
document, so they should name the same model.

Needs atomate2 (``pip install "atombatch[atomate2]"``).
"""

import copy


def atomate2_task(maker, structure, model=None):
    """Return a task that runs ``maker`` on ``structure`` with batched evaluations.

    ``maker`` is an atomate2 maker with a ``run_ase`` method and a job
    ``make`` (for example ``ForceFieldRelaxMaker(relax_cell=True)``), and
    ``structure`` a pymatgen ``Structure`` or ``Molecule``. The job runs
    directly, not as a jobflow ``Job``, and the task's value is its document,
    such as a ``ForceFieldTaskDocument``. ``model`` names the atombatch model
    (default: the model the task is submitted with).
    """
    make = getattr(type(maker).make, "original", None)
    if make is None or not hasattr(maker, "run_ase"):
        raise TypeError(
            f"{type(maker).__name__} is not an atomate2 maker that runs through ASE "
            "(it needs run_ase and a job make method)."
        )
    # atomate2's trajectory observer reads stress at every step of periodic
    # systems, so periodic structures request it with energy and forces.
    periodic = hasattr(structure, "lattice")
    properties = ("energy", "forces", "stress") if periodic else ("energy", "forces")

    def task(context):
        job_maker = copy.copy(maker)
        job_maker._calculator = context.calculator(model, properties=properties)
        job_maker.run_ase = _quiet(job_maker)
        return make(job_maker, structure)

    return task


def _quiet(maker):
    """Return ``maker.run_ase`` that leaves ``sys.stdout`` alone.

    atomate2 silences the optimizer log with ``contextlib.redirect_stdout``,
    which replaces the process-wide ``sys.stdout``. Relaxations in concurrent
    task threads then restore each other's buffers and can leave ``sys.stdout``
    pointing at one, so that later output is lost. During the relaxation, the
    optimizer log goes to no file (``logfile=None``) and atomate2 redirects
    stdout to itself (``verbose=True``); the document still records the
    maker's own settings.
    """
    run_ase = maker.run_ase

    def quiet_run_ase(*args, **kwargs):
        settings = maker.relax_kwargs
        maker.relax_kwargs = {"logfile": None, **settings, "verbose": True}
        try:
            return run_ase(*args, **kwargs)
        finally:
            maker.relax_kwargs = settings

    return quiet_run_ase
